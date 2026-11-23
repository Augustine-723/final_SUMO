"""Convert one NCHU gate episode of complete quarter-hour counts into SUMO demand."""
import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ARMS = "NESW"
FIELDS = ["site", "basis", "source", "survey_start", "start", "arm", "exit", "mode", "count"]
# A pedestrian crosses the named arm. Motor traffic enters arm and leaves exit.
KEYS = {(a, b, mode) for a in ARMS for b in ARMS if b != a for mode in ("car", "motorcycle")}
KEYS.update((a, a, "pedestrian") for a in ARMS)


def load_counts(path):
    bins = {}
    provenance = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["site"] != "nchu_gate" or row["basis"] not in ("synthetic", "observed") or not row["source"].strip():
                raise ValueError("Missing NCHU source or invalid basis")
            if row["basis"] == "observed":
                stamp = datetime.fromisoformat(row["survey_start"])
                if stamp.utcoffset() != timedelta(hours=8) or stamp.minute % 15 or stamp.second or stamp.microsecond:
                    raise ValueError("Use a quarter-hour episode start with Taiwan UTC+08:00")
            provenance.add((row["basis"], row["source"], row["survey_start"]))
            start = int(row["start"])
            key = (row["arm"], row["exit"], row["mode"])
            bucket = bins.setdefault(start, {})
            count = int(row["count"])
            if start < 0 or start % 900 or key not in KEYS or key in bucket or count < 0:
                raise ValueError(f"Invalid or duplicate count: {row}")
            bucket[key] = count
    if len(provenance) != 1 or sorted(bins) != list(range(0, len(bins) * 900, 900)):
        raise ValueError("Require a single documented source and contiguous bins starting at zero")
    for start, bucket in bins.items():
        if set(bucket) != KEYS:
            raise ValueError(f"Incomplete interval {start}; unknown counts must not be filled as zero")
    return bins


def write_xml(root, path):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def routes(bins, seed, path, network):
    rng = random.Random(seed)
    root = ET.Element("routes")
    for name, vclass in (("car", "passenger"), ("motorcycle", "motorcycle")):
        ET.SubElement(root, "vType", id=name, vClass=vclass)
    events = []
    for start, bucket in sorted(bins.items()):
        for key, count in sorted(bucket.items()):
            for _ in range(count):
                events.append((start + rng.randrange(900), key))
    net = ET.parse(network)
    for i, (depart, (arm, dest, mode)) in enumerate(sorted(events)):
        args = {"id": f"{mode}_{i}", "depart": str(depart)}
        if mode == "pedestrian":
            length = float(net.find(f".//edge[@id='{arm}_in']/lane").get("length"))
            person = ET.SubElement(root, "person", **args, departPos=str(max(1, length - 4)))
            ET.SubElement(person, "walk", edges=f"{arm}_in {arm}_out", arrivalPos="4")
        else:
            vehicle = ET.SubElement(root, "vehicle", **args, type=mode)
            ET.SubElement(vehicle, "route", edges=f"{arm}_in {dest}_out")
    write_xml(root, path)


def synthetic(name, path):
    # Engineering test loads, per arm per 15 min; no site observations are embedded.
    profiles = {"offpeak": ([18, 20, 8, 20], [4, 5]),
                "morning": ([42, 38, 12, 40], [12, 18]),
                "evening": ([25, 42, 34, 45], [16, 12])}
    base, walkers = profiles[name]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(FIELDS)
        for slot in range(2):
            for idx, arm in enumerate(ARMS):
                prefix = ["nchu_gate", "synthetic", f"{name}_v1", "", slot * 900, arm]
                writer.writerow([*prefix, arm, "pedestrian", walkers[slot]])
                total = base[idx] + slot * 4
                # N opposite S, E opposite W. Split through, right, then remainder left.
                right, through = total // 4, total // 2
                for offset, volume in ((3, right), (2, through), (1, total - through - right)):
                    dest = ARMS[(idx + offset) % 4]
                    bikes = volume // 3
                    writer.writerow([*prefix, dest, "car", volume - bikes])
                    writer.writerow([*prefix, dest, "motorcycle", bikes])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for name in ("offpeak", "morning", "evening"):
        synthetic(name, args.directory / f"{name}_synthetic.csv")
