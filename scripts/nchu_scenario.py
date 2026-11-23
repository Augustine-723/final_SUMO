"""Build the NCHU junction and run a count-conserving fixed-time simulation."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

from nchu_demand import ARMS, ROOT, load_counts, routes, write_xml


def network(out):
    nodes = ET.Element("nodes")
    edges = ET.Element("edges")
    crossings = ET.Element("connections")
    ET.SubElement(nodes, "node", id="gate", x="0", y="0", type="traffic_light")
    shapes = {a: [] for a in ARMS}
    with (ROOT / "sumo/nchu/geometry.csv").open() as stream:
        for row in csv.DictReader(stream):
            lat, lon = float(row["latitude"]), float(row["longitude"])
            x = math.radians(lon - 120.6751494) * 6371000 * math.cos(math.radians(24.1241218))
            y = math.radians(lat - 24.1241218) * 6371000
            point = (f"{x:.2f}", f"{y:.2f}")
            points = shapes[row["arm"]]
            if not points or points[-1] != point:
                points.append(point)
    for arm, points in shapes.items():
        x, y = points[-1]
        ET.SubElement(nodes, "node", id=arm, x=x, y=y)
        for side, origin, dest, coords in (("in", arm, "gate", points[::-1]), ("out", "gate", arm, points)):
            ET.SubElement(edges, "edge", {"id": f"{arm}_{side}", "from": origin, "to": dest,
                          "numLanes": "1", "sidewalkWidth": "2", "speed": "8.33",
                          "shape": " ".join(",".join(p) for p in coords)})
        ET.SubElement(crossings, "crossing", node="gate", edges=f"{arm}_in {arm}_out", width="3")
    for element, name in ((nodes, "nodes.xml"), (edges, "edges.xml"), (crossings, "connections.xml")):
        write_xml(element, out / name)
    subprocess.run(["netconvert", "-n", "nodes.xml", "-e", "edges.xml", "-x", "connections.xml",
                    "--no-turnarounds", "--tls.green.time", "25", "--tls.yellow.time", "3",
                    "--tls.allred.time", "2", "-o", "gate.net.xml"], cwd=out, check=True)
    net = ET.parse(out / "gate.net.xml")
    links = {(c.get("from"), c.get("to")) for c in net.findall("connection")}
    if any((f"{a}_in", f"{b}_out") not in links for a in ARMS for b in ARMS if a != b):
        raise ValueError("Missing turning movement")
    if len(net.findall("tlLogic")) != 1 or len(net.findall("edge[@function='crossing']")) != 4:
        raise ValueError("Expected one controller and four crossings")


def verify_crossings(out):
    net = ET.parse(out / "gate.net.xml")
    crossings = {e.get("id"): e.get("crossingEdges").split()[0].split("_")[0]
                 for e in net.findall("edge[@function='crossing']")}
    people = ET.parse(out / "gate.rou.xml").findall("person")
    expected = {p.get("id"): p.find("walk").get("edges").split("_")[0] for p in people}
    seen = set()
    with gzip.open(out / "trace.xml.gz") as trace:
        for _, step in ET.iterparse(trace):
            if step.tag != "timestep":
                continue
            for p in step.findall("person"):
                arm = crossings.get(p.get("edge"))
                if arm:
                    if expected[p.get("id")] != arm:
                        raise ValueError("Pedestrian used a different crossing")
                    seen.add(p.get("id"))
            step.clear()
    if seen != set(expected):
        raise ValueError("Missing pedestrian crossing traversal")
    return len(seen)


def simulate(counts, out, seed, gui):
    bins = load_counts(counts)
    out.mkdir(parents=True, exist_ok=True)
    network(out)
    routes(bins, seed, out / "gate.rou.xml", out / "gate.net.xml")
    config = ET.Element("configuration")
    inputs = ET.SubElement(config, "input")
    ET.SubElement(inputs, "net-file", value="gate.net.xml")
    ET.SubElement(inputs, "route-files", value="gate.rou.xml")
    write_xml(config, out / "gate.sumocfg")
    cmd = ["sumo-gui" if gui else "sumo", "-c", "gate.sumocfg", "--seed", str(seed),
           "--end", str(len(bins) * 900 + 900), "--tripinfo-output", "trips.xml",
           "--fcd-output", "trace.xml.gz",
           "--collision.action", "warn", "--statistic-output", "statistics.xml",
           "--no-step-log", "true"]
    if gui:
        cmd.extend(["--start", "--quit-on-end"])
    subprocess.run(cmd, cwd=out, check=True)
    trips = ET.parse(out / "trips.xml")
    completed = Counter(t.get("vType") for t in trips.findall("tripinfo"))
    people = trips.findall("personinfo")
    completed["pedestrian"] = sum(float(p.find("walk").get("arrival")) >= 0 for p in people)
    expected = Counter()
    for bucket in bins.values():
        for (_, _, mode), count in bucket.items():
            expected[mode] += count
    stats = ET.parse(out / "statistics.xml")
    with counts.open(encoding="utf-8-sig") as stream:
        origin = next(csv.DictReader(stream))
    result = {"basis": origin["basis"], "source": origin["source"],
              "input_sha256": hashlib.sha256(counts.read_bytes()).hexdigest(),
              "verified_crossings": verify_crossings(out),
              "vehicle_wait_mean_s": float(stats.find("vehicleTripStatistics").get("waitingTime")),
              "seed": seed, "input": counts.name, "expected": dict(expected),
              "completed": dict(completed), "teleports": int(stats.find("teleports").get("total")),
              "person_teleports": int(stats.find("personTeleports").get("total")) if stats.find("personTeleports") is not None else 0,
              "collisions": int(stats.find("safety").get("collisions"))}
    if completed != expected or result["teleports"] or result["collisions"] or result["person_teleports"]:
        raise RuntimeError(f"Failed scenario checks: {result}")
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("counts", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    simulate(args.counts, args.out, args.seed, args.gui)
