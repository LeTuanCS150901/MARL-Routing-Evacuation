"""Evacuation metrics of one finished run, from SUMO's tripinfo.xml.

Usage: python metrics.py <run folder or tripinfo.xml> [--demand <file.rou.xml>]
Writes metrics.json next to the run's sumo/ folder and prints one line.

Metrics (all in seconds of simulated time):
  cars       cars in the demand file (or loaded cars if the demand file is not given)
  arrived    cars that reached a shelter; arrival_pct = arrived / cars
  mean_trip, median_trip, min_trip, max_trip   trip time of the arrived cars (arrival - departure)
  last_arrival   time the last car arrived (network clearance time)
  per_origin     the same numbers for each start road (= group)
"""
import argparse
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path


def count_demand(route_file):
    n = 0
    for _, el in ET.iterparse(route_file):
        if el.tag in ("vehicle", "trip"):
            n += 1
        el.clear()
    return n


def summary(trips):
    if not trips:
        return {"arrived": 0}
    d = [t for _, t, _ in trips]
    return {"arrived": len(d), "mean_trip": round(statistics.mean(d), 1), "median_trip": round(statistics.median(d), 1),
            "min_trip": round(min(d), 1), "max_trip": round(max(d), 1), "last_arrival": round(max(a for _, _, a in trips), 1)}


def compute(path, demand=None):
    path = Path(path)
    tripinfo = path if path.suffix == ".xml" else path / "sumo" / "tripinfo.xml"
    trips = []
    for _, el in ET.iterparse(tripinfo):
        if el.tag == "tripinfo":
            origin = el.get("departLane", "").rsplit("_", 1)[0]
            trips.append((origin, float(el.get("duration")), float(el.get("arrival"))))
            el.clear()
    cars = count_demand(demand) if demand else None
    out = {"run": str(tripinfo.parent.parent if tripinfo.parent.name == "sumo" else tripinfo.parent), "cars": cars, **summary(trips)}
    if cars:
        out["arrival_pct"] = round(100.0 * len(trips) / cars, 2)
    out["per_origin"] = {o: summary([t for t in trips if t[0] == o]) for o in sorted({t[0] for t in trips})}
    return out, tripinfo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--demand", default=None)
    a = ap.parse_args()
    m, tripinfo = compute(a.path, a.demand)
    target = (tripinfo.parent.parent if tripinfo.parent.name == "sumo" else tripinfo.parent) / "metrics.json"
    target.write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(f"arrived {m['arrived']}/{m['cars'] if m['cars'] else '?'}  mean {m.get('mean_trip')} s  median {m.get('median_trip')} s  "
          f"min {m.get('min_trip')} s  max {m.get('max_trip')} s  last arrival {m.get('last_arrival')} s  -> {target}")
