"""
Regenerate the auto-managed sections of README.md.

Reads:
  - data/results/{year}_*.json          per-race prediction + actuals + metrics
  - data/results/upcoming_prediction.json  the next race's prediction or stub
  - FastF1 schedule (for round-number lookups)

Writes between marker pairs:
  <!-- accuracy-start --> ... <!-- accuracy-end -->
  <!-- next-race-start --> ... <!-- next-race-end -->

Idempotent. Leaves everything else in the README untouched.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RESULTS_DIR = ROOT / "data" / "results"
UPCOMING_FILE = RESULTS_DIR / "upcoming_prediction.json"

ACCURACY_START = "<!-- accuracy-start -->"
ACCURACY_END = "<!-- accuracy-end -->"
NEXT_START = "<!-- next-race-start -->"
NEXT_END = "<!-- next-race-end -->"


# ---------------------------------------------------------------------------
# Schedule lookup (round numbers + next upcoming race name)
# ---------------------------------------------------------------------------
def load_schedule(year: int) -> dict:
    """
    Build {lower_event_name: {round, date}} for the season, plus the
    next-upcoming race name.

    FastF1 isn't always available (e.g. running locally without internet);
    fall back to empty info on failure.
    """
    info: dict = {"by_name": {}, "next_upcoming": None}
    try:
        import fastf1
        import pandas as pd

        # Match scripts/race_update.py's cache location.
        cache_dir = ROOT / "data" / "raw"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))

        sched = fastf1.get_event_schedule(year, include_testing=False)
        date_col = "Session5DateUtc" if "Session5DateUtc" in sched.columns else "EventDate"
        sched["RaceEndUtc"] = pd.to_datetime(sched[date_col], utc=True)

        for _, row in sched.iterrows():
            info["by_name"][str(row["EventName"]).lower()] = {
                "round": int(row["RoundNumber"]),
                "date": row["RaceEndUtc"],
            }

        now = pd.Timestamp.now(tz="UTC")
        upcoming = sched[sched["RaceEndUtc"] >= now]
        if len(upcoming):
            row = upcoming.iloc[0]
            info["next_upcoming"] = {
                "name": str(row["EventName"]),
                "round": int(row["RoundNumber"]),
                "date": row["RaceEndUtc"],
            }
    except Exception as exc:
        print(f"[update_readme] schedule lookup failed ({exc}) — round numbers will be blank.")

    return info


def lookup_round(schedule_info: dict, race_name: str) -> str:
    entry = schedule_info["by_name"].get(race_name.lower())
    return str(entry["round"]) if entry else ""


# ---------------------------------------------------------------------------
# Per-race results (Live model accuracy)
# ---------------------------------------------------------------------------
def load_season(year: int) -> list[dict]:
    if not RESULTS_DIR.exists():
        return []
    out: list[dict] = []
    for path in sorted(RESULTS_DIR.glob(f"{year}_*.json")):
        try:
            out.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def rating_label(mean_spearman: float) -> str:
    if mean_spearman >= 0.7:
        return "STRONG"
    if mean_spearman >= 0.5:
        return "DECENT"
    return "NEEDS WORK"


def render_accuracy_section(year: int, schedule_info: dict) -> str:
    races = load_season(year)
    scored = [r for r in races if "spearman" in r]

    # Sort by round number (so chronological — accuracy_tracker filenames
    # are alphabetic which mis-orders 'Australian' before 'Bahrain', etc.)
    scored.sort(key=lambda r: int(lookup_round(schedule_info, r["race_name"]) or 999))

    lines: list[str] = []
    lines.append("## Live model accuracy")
    lines.append("")
    lines.append(
        "_Auto-updated by [.github/workflows/race-update.yml]"
        "(.github/workflows/race-update.yml). Pre-race prediction generated "
        "Saturday 23:00 UTC after qualifying; race scored Monday 12:00 UTC._"
    )
    lines.append("")

    if not scored:
        lines.append(f"_No {year} races scored yet — check back after the next round._")
        return "\n".join(lines)

    avg_sp = mean(r["spearman"] for r in scored)
    avg_t3 = mean(r.get("top3_correct", 0) for r in scored)
    latest = scored[-1]

    lines.append(f"### {year} season — {len(scored)} race(s) scored")
    lines.append("")
    lines.append(f"**Latest:** {latest['race_name']} — Spearman **{latest['spearman']:.3f}**, top-3 **{latest.get('top3_correct', 0)}/3**.")
    lines.append("")
    lines.append("| Mean Spearman | Mean Top-3 (out of 3) | Rating |")
    lines.append("|---|---|---|")
    lines.append(f"| {avg_sp:.3f} | {avg_t3:.2f} | {rating_label(avg_sp)} |")
    lines.append("")

    lines.append("### Per-race results")
    lines.append("")
    lines.append("| Round | Race | Spearman | Top-3 | Predicted P1 → P3 | Actual P1 → P3 |")
    lines.append("|---|---|---|---|---|---|")
    for r in scored:
        round_num = lookup_round(schedule_info, r["race_name"]) or "—"
        race = r["race_name"]
        sp = r["spearman"]
        t3 = r.get("top3_correct", 0)
        pred_top3 = " → ".join(_top_n_drivers(r["predictions"], "PredictedPosition", 3))
        actual_top3 = " → ".join(_top_n_drivers(r.get("actuals", []), "FinishPosition", 3))
        lines.append(f"| {round_num} | {race} | {sp:.3f} | {t3}/3 | {pred_top3} | {actual_top3} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Next race prediction
# ---------------------------------------------------------------------------
def render_next_race_section(year: int, schedule_info: dict) -> str:
    """
    Three states this section can be in:

    1. upcoming_prediction.json has predictions AND its race is still in the
       future → render full prediction table.
    2. upcoming_prediction.json's race date is in the past → that prediction
       is now stale (the race is already scored above). Fall back to
       schedule's next_upcoming and show a placeholder.
    3. No upcoming file or empty predictions → placeholder.
    """
    next_upcoming = schedule_info.get("next_upcoming")

    data = None
    if UPCOMING_FILE.exists():
        try:
            data = json.loads(UPCOMING_FILE.read_text())
        except json.JSONDecodeError:
            data = None

    if data:
        # Is the predicted race still in the future? If not, treat as stale.
        try:
            import pandas as pd
            race_date = pd.to_datetime(data.get("race_date_utc"), utc=True)
            now = pd.Timestamp.now(tz="UTC")
            stale = race_date < now
        except Exception:
            stale = False

        if stale:
            data = None  # fall through to placeholder

    lines: list[str] = ["## Next race prediction", ""]

    # State 3: no data at all and we couldn't find a next race
    if not data and not next_upcoming:
        lines.append("_No upcoming race on the schedule._")
        return "\n".join(lines)

    # State 2: stale (or missing) — show placeholder for the actual next race
    if not data and next_upcoming:
        date_str = next_upcoming["date"].strftime("%Y-%m-%d")
        lines.append(f"**{year} {next_upcoming['name']} — Round {next_upcoming['round']}** ({date_str})")
        lines.append("")
        lines.append("_Prediction will appear here after qualifying._")
        return "\n".join(lines)

    # State 1: live prediction
    race = data.get("race_name", "Unknown")
    round_num = data.get("round", "")
    preds = data.get("predictions", [])
    top10 = sorted(preds, key=lambda p: p.get("PredictedPosition", 999))[:10]

    lines.append(f"**{year} {race} — Round {round_num}**")
    lines.append("")

    if not top10:
        lines.append("_Prediction will appear here after qualifying._")
        return "\n".join(lines)

    has_conf = any("Confidence" in p for p in top10)
    has_team = any("TeamName" in p for p in top10)

    header_cols = ["Predicted", "Driver"]
    if has_team:
        header_cols.append("Team")
    if has_conf:
        header_cols.append("Confidence")

    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")

    for p in top10:
        cells = [str(p.get("PredictedPosition", "")), str(p.get("Abbreviation", ""))]
        if has_team:
            cells.append(str(p.get("TeamName", "")))
        if has_conf:
            cells.append(f"{p.get('Confidence', 0):.2f}")
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _top_n_drivers(rows: list[dict], pos_key: str, n: int) -> list[str]:
    rows_sorted = sorted(
        (r for r in rows if r.get(pos_key) is not None),
        key=lambda r: r[pos_key],
    )
    return [str(r.get("Abbreviation", "?")) for r in rows_sorted[:n]]


def replace_section(readme: str, start: str, end: str, body: str) -> str:
    pattern = re.compile(
        re.escape(start) + r".*?" + re.escape(end),
        flags=re.DOTALL,
    )
    block = f"{start}\n{body}\n{end}"
    if pattern.search(readme):
        return pattern.sub(block, readme)
    return readme.rstrip() + "\n\n" + block + "\n"


def main() -> int:
    if not README.exists():
        print("[update_readme] README.md not found.")
        return 1

    year = dt.datetime.now(dt.timezone.utc).year
    schedule_info = load_schedule(year)

    text = README.read_text(encoding="utf-8")
    text = replace_section(text, ACCURACY_START, ACCURACY_END, render_accuracy_section(year, schedule_info))
    text = replace_section(text, NEXT_START, NEXT_END, render_next_race_section(year, schedule_info))
    README.write_text(text, encoding="utf-8")
    print("[update_readme] README.md updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
