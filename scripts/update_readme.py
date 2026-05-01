"""
Regenerate the auto-managed sections of README.md.

Reads:
  - data/results/{year}_*.json  (per-race prediction + actuals + metrics)
  - data/results/upcoming_prediction.json (next race's prediction)

Writes inside README.md, between marker pairs:
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


def render_accuracy_section(year: int) -> str:
    races = load_season(year)
    scored = [r for r in races if "spearman" in r]

    lines: list[str] = []
    lines.append("## Live model accuracy")
    lines.append("")
    lines.append(
        f"_Auto-updated by [.github/workflows/race-update.yml]"
        f"(.github/workflows/race-update.yml) every Monday after each race._"
    )
    lines.append("")

    if not scored:
        lines.append(f"_No {year} races scored yet — check back after the next round._")
        return "\n".join(lines)

    avg_sp = mean(r["spearman"] for r in scored)
    avg_t3 = mean(r.get("top3_correct", 0) for r in scored)

    lines.append(f"### {year} season — {len(scored)} race(s) scored")
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
        round_num = _round_from_filename(r)
        race = r["race_name"]
        sp = r["spearman"]
        t3 = r.get("top3_correct", 0)
        pred_top3 = " → ".join(_top_n_drivers(r["predictions"], "PredictedPosition", 3))
        actual_top3 = " → ".join(_top_n_drivers(r.get("actuals", []), "FinishPosition", 3))
        lines.append(f"| {round_num} | {race} | {sp:.3f} | {t3}/3 | {pred_top3} | {actual_top3} |")

    return "\n".join(lines)


def render_next_race_section() -> str:
    if not UPCOMING_FILE.exists():
        return "## Next race prediction\n\n_No upcoming race scheduled, or prediction not yet generated._"

    data = json.loads(UPCOMING_FILE.read_text())
    race = data.get("race_name", "Unknown")
    year = data.get("year", "")
    round_num = data.get("round", "")
    preds = data.get("predictions", [])
    top10 = sorted(preds, key=lambda p: p.get("PredictedPosition", 999))[:10]

    lines: list[str] = []
    lines.append("## Next race prediction")
    lines.append("")
    lines.append(f"**{year} {race} — Round {round_num}**")
    lines.append("")

    if not top10:
        lines.append("_Prediction not yet generated._")
        return "\n".join(lines)

    has_conf = any("Confidence" in p for p in top10)
    if has_conf:
        lines.append("| Predicted | Driver | Confidence |")
        lines.append("|---|---|---|")
        for p in top10:
            lines.append(f"| {p.get('PredictedPosition', '')} | {p.get('Abbreviation', '')} | {p.get('Confidence', 0):.2f} |")
    else:
        lines.append("| Predicted | Driver |")
        lines.append("|---|---|")
        for p in top10:
            lines.append(f"| {p.get('PredictedPosition', '')} | {p.get('Abbreviation', '')} |")

    return "\n".join(lines)


def _round_from_filename(entry: dict) -> str:
    # accuracy_tracker stores files as {year}_{race_slug}.json — round is not
    # in the JSON itself, so we infer it from order. Fallback: blank.
    return entry.get("round", "")


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
    # First-time insertion: append at end of README.
    return readme.rstrip() + "\n\n" + block + "\n"


def main() -> int:
    if not README.exists():
        print("[update_readme] README.md not found.")
        return 1

    year = dt.datetime.now(dt.timezone.utc).year
    text = README.read_text(encoding="utf-8")
    text = replace_section(text, ACCURACY_START, ACCURACY_END, render_accuracy_section(year))
    text = replace_section(text, NEXT_START, NEXT_END, render_next_race_section())
    README.write_text(text, encoding="utf-8")
    print("[update_readme] README.md updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
