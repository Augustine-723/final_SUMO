"""Launch the NCHU fixed-time demo with an orthophoto and detailed vehicle bodies."""
import argparse
import math
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen
from xml.etree import ElementTree as ET

from nchu_demand import ROOT, load_counts, routes, write_xml
from nchu_scenario import network

# EPSG:4326 bounds in WMS 1.1.1 longitude, latitude order.
BOUNDS = (120.6744, 24.1236, 120.6759, 24.1247)
PHOTO = ("https://wms.nlsc.gov.tw/wms?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap"
         "&LAYERS=PHOTO2&STYLES=&SRS=EPSG:4326&BBOX=120.6744,24.1236,120.6759,24.1247"
         "&WIDTH=768&HEIGHT=600&FORMAT=image/jpeg")
PALETTE = ("226,229,223", "56,82,112", "186,67,47", "80,84,91", "165,175,182", "240,222,177")

def prepare(out, counts, seed):
    out.mkdir(parents=True, exist_ok=True)
    network(out)
    net = out / "gate.net.xml"
    route_file = out / "gate.rou.xml"
    bins = load_counts(counts)
    routes(bins, seed, route_file, net)
    traffic = ET.parse(route_file).getroot()
    for t in traffic.findall("vType"):
        t.set("guiShape", "passenger/sedan" if t.get("id") == "car" else "motorcycle")
    for i, vehicle in enumerate(traffic.findall("vehicle")):
        vehicle.set("color", PALETTE[i % len(PALETTE)])
    for person in traffic.findall("person"):
        person.set("color", "225,73,48")
    write_xml(traffic, route_file)

    junction = ET.parse(net).find("junction[@id='gate']")
    x, y = float(junction.get("x")), float(junction.get("y"))
    view = ET.Element("viewsettings")
    scheme = ET.SubElement(view, "scheme", name="NCHU main gate")
    ET.SubElement(scheme, "background", backgroundColor="186,193,176")
    ET.SubElement(scheme, "vehicles", vehicleMode="0", vehicleQuality="2", showBlinker="1")
    ET.SubElement(scheme, "persons", personMode="0", personQuality="2")
    ET.SubElement(view, "viewport", x=str(x), y=str(y), zoom="880")
    photo = out / "nchu-photo2.jpg"
    if not photo.exists():
        try:
            with urlopen(PHOTO, timeout=18) as response:
                data = response.read()
            if not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
                raise ValueError("Expected a JPEG orthophoto")
            photo.write_bytes(data)
        except (OSError, ValueError) as exc:
            print(f"NLSC orthophoto unavailable ({exc}); using plain background.", file=sys.stderr)
    if photo.exists():
        west, south, east, north = BOUNDS
        m_lat = math.pi * 6371000 / 180
        m_lon = m_lat * math.cos(math.radians(24.1241218))
        ET.SubElement(view, "decal", file=photo.name,
                      centerX=str(x + ((east+west)/2 - 120.6751494)*m_lon),
                      centerY=str(y + ((north+south)/2 - 24.1241218)*m_lat),
                      width=str((east-west)*m_lon), height=str((north-south)*m_lat), layer="-100")
    write_xml(view, out / "view.xml")
    cfg = ET.Element("configuration")
    inputs = ET.SubElement(cfg, "input")
    ET.SubElement(inputs, "net-file", value=net.name)
    ET.SubElement(inputs, "route-files", value=route_file.name)
    gui = ET.SubElement(cfg, "gui_only")
    ET.SubElement(gui, "gui-settings-file", value="view.xml")
    ET.SubElement(gui, "delay", value="100")
    ET.SubElement(ET.SubElement(cfg,"time"),"end",value=str(900*(len(bins)+1)))
    ET.SubElement(ET.SubElement(cfg,"random_number"),"seed",value=str(seed))
    write_xml(cfg, out / "gate.sumocfg")
    print("Basemap: NLSC PHOTO2 (when available). Road geometry: OpenStreetMap contributors.", flush=True)
    return out / "gate.sumocfg"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"results/nchu/showcase")
    parser.add_argument("--counts", type=Path, default=ROOT/"data/nchu/morning_synthetic.csv")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--prepare-only",action="store_true")
    args = parser.parse_args()
    config = prepare(args.out, args.counts, args.seed)
    if not args.prepare_only:
        subprocess.run(["sumo-gui","-c",str(config),"--start"],check=True)
