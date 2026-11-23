"""Rebuild EI035 survey counts and 15-minute scenario inputs from the verified workbook."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = dict(zip("ABCD", ("east", "south", "west", "north")))
# D:P in the raw survey. Two-stage turns are a separate reporting category.
CLASSES = ("heavy", "bus", "light", "motorcycle")
MOVEMENTS = ([("left", c) for c in CLASSES] + [("two_stage_left", "motorcycle")]
             + [("through", c) for c in CLASSES] + [("right", c) for c in CLASSES])
HOURS = (7, 8, 17, 18)


def table(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def scaled_counts(counts, numerator, denominator):
    """Round the bin total half-up, then distribute the remainder by original column order."""
    floors = [v * numerator // denominator for v in counts]
    target = (2 * sum(counts) * numerator + denominator) // (2 * denominator)
    ranking = sorted(range(len(counts)), key=lambda i: -(counts[i] * numerator % denominator))
    for idx in ranking[:target - sum(floors)]:
        floors[idx] += 1
    return floors


def build(source, workbook, out):
    if hashlib.sha256(workbook.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("Workbook SHA-256 differs from the source manifest")
    book = load_workbook(workbook, read_only=True, data_only=True)
    try:
        raw = list(book["調查"].values)
        hourly = list(book["表"].values)
        if book["基本"]["B6"].value.date().isoformat() != source["survey_date"]:
            raise ValueError("Survey date mismatch")
    finally:
        book.close()
    if len(raw) != 68 or len(raw[0]) != 56 or hourly[0][2] != "EI035":
        raise ValueError("Unexpected EI035 workbook layout")
    audit, demand = [], []
    totals, by_hour, statuses, episodes = Counter(), Counter(), Counter(), Counter()
    group = movement = ""
    headers = []
    for col in range(3, 56):
        group = raw[0][col] or group
        movement = raw[1][col] or movement
        headers.append((group, movement.replace("\n", ""), raw[2][col] or ""))
    for index, row in enumerate(raw[4:]):
        block, slot = divmod(index, 16)
        arm = "ABCD"[block]
        hour = HOURS[slot // 4]
        minute = slot % 4 * 15
        start, end = f"{hour:02}:{minute:02}", f"{hour + (minute == 45):02}:{(minute + 15) % 60:02}"
        if (row[0] != (arm if slot == 0 else None) or tuple(row[1:3]) != (start, end)):
            raise ValueError(f"Unexpected approach/time at survey row {index + 5}")
        for col, header in enumerate(headers, 3):
            val = row[col]
            if val is not None and (type(val) is not int or val < 0):
                raise ValueError(f"Invalid count at row {index + 5}, column {col + 1}")
            status = "blank" if val is None else ("zero" if val == 0 else "positive")
            statuses[status] += 1
            audit.append((source["survey_date"], arm, start, end, *header, val, status,
                          f"{get_column_letter(col + 1)}{index + 5}"))
        raw_counts = row[3:16]
        counts = [v or 0 for v in raw_counts]  # Numerical subtotal over reported cells only.
        peak = "am" if hour < 12 else "pm"
        for (turn, kind), val in zip(MOVEMENTS, counts):
            by_hour[arm, hour, turn, kind] += val
            totals[peak, "two_stage" if turn == "two_stage_left" else "ordinary"] += val
        totals[peak, "bicycle"] += row[55]
        scenarios = [(peak + "_observed", counts)]
        if peak == "am":
            scenarios += [("offpeak_synthetic", scaled_counts(counts, 1, 2)),
                          ("directional_synthetic", scaled_counts(counts, 3 if arm == "A" else 2, 2))]
        for scenario, values in scenarios:
            for (turn, kind), value, original in zip(MOVEMENTS, values, raw_counts):
                if original is None:
                    continue
                demand.append((scenario, source["survey_date"], start, end, (slot % 8) * 900,
                               arm, DIRECTIONS[arm], turn, kind, value))
                episodes[scenario] += value

    # Cross-check 16 official hourly rows: heavy+bus, light, motorcycle for each
    # of three turns, plus separately reported two-stage turns. Compare counts, never PCU.
    checks = 0
    for block, arm in enumerate("ABCD"):
        for i, hour in enumerate(HOURS):
            ref = hourly[9 + 5 * block + i]
            if ref[1] != f"{hour:02}:00-{hour + 1:02}:00":
                raise ValueError("Unexpected official hourly summary")
            for turn, col in (("left", 3), ("through", 10), ("right", 16)):
                quantities = [by_hour[arm, hour, turn, c] for c in CLASSES]
                if [quantities[0] + quantities[1], *quantities[2:]] != list(ref[col:col + 3]):
                    raise ValueError(f"Hourly reconciliation failed: {arm}, {hour}, {turn}")
                checks += 3
            if by_hour[arm, hour, "two_stage_left", "motorcycle"] != ref[8]:
                raise ValueError(f"Two-stage reconciliation failed: {arm}, {hour}")
            checks += 1
    table(out / "ei035_survey.csv", ["survey_date","approach","start","end","lane_group","movement_header",
          "subheader","count","status","source_cell"], audit)
    table(out / "ei035_demand.csv", ["scenario","source_date","source_start","source_end","episode_start_s",
          "approach","approach_side","movement","vehicle_class","count_15min"], demand)
    summary = {"station":"EI035","survey_date":source["survey_date"],"source_sha256":source["sha256"],
               "approach_intervals":len(raw)-4,"cell_status":dict(statuses),"official_hourly_checks":checks,
               "observed_counts":{p:{k:totals[p,k] for k in ("ordinary","two_stage","bicycle")} for p in ("am","pm")},
               "scenario_movement_counts_including_two_stage":dict(episodes)}
    (out / "ei035_quality.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, help="Local copy of official 114 EI035.xlsx")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()
    with (ROOT / "data/metadata/site_sources.csv").open() as stream:
        source = next(r for r in csv.DictReader(stream) if (r["roc_year"],r["station"]) == ("114","EI035"))
    args.output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output) as tmp:
        stage = Path(tmp)
        workbook = args.workbook
        if workbook is None:
            workbook = stage / "EI035.xlsx"
            with urlopen(source["url"], timeout=90) as response:
                workbook.write_bytes(response.read())
        build(source, workbook, stage)
        for name in ("ei035_survey.csv", "ei035_demand.csv", "ei035_quality.json"):
            (stage / name).replace(args.output / name)
    print((args.output / "ei035_quality.json").read_text())


if __name__ == "__main__":
    main()
