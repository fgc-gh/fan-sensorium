#!/usr/bin/env python3
"""
Neighborhood Sensorium :: contract.

The gate between "the pipeline ran" and "the public map changes". If any check
fails, the run fails, nothing is published, and last week's map stays up.
That is the whole maintenance plan: a red X in the Actions tab and an email,
instead of a quietly wrong map in front of city council.

Checks the failure modes that actually happen with public GIS services:
  * the city renames or splits the neighbourhood polygon (0 or 2+ features)
  * a server returns coordinates in the wrong projection (lots land in Idaho)
  * a feed silently truncates (taxlot count drops by a third)
  * a heat or green-cover grid built from too few or nonsense scenes

With --write, a passing run records counts and checksums in data/manifest.json.
Committed weekly, that file's git history is a log of how the data moved.

Fertile Ground Conservancy.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path

# Generous envelope around the City of Eugene, lon/lat.
EUGENE = (-123.30, 43.95, -122.90, 44.15)
BOUNDARY_KM2 = (0.5, 25.0)
TAXLOTS_MIN = 300
TAXLOT_SWING = 0.15          # max week-over-week change in count
TAXLOT_ID_COVERAGE = 0.95
HEAT_MIN_SCENES = 3
HEAT_MEDIAN_F = (50.0, 140.0)  # plausible summer surface temps
HEAT_MAX_ABS_DELTA = 30.0
HEAT_LOT_COVERAGE = 0.95
GREEN_MIN_SCENES = 3
GREEN_MEDIAN_NDVI = (0.05, 0.9)  # a summer residential neighbourhood, not desert or forest
GREEN_LOT_COVERAGE = 0.95


class ContractError(Exception):
    pass


def _coords(geom: dict):
    def walk(c):
        if isinstance(c[0], (int, float)):
            yield (c[0], c[1])
        else:
            for x in c:
                yield from walk(x)
    yield from walk(geom["coordinates"])


def ring_area_km2(ring: list) -> float:
    """Spherical-excess-free approximation, fine at neighbourhood scale."""
    lat0 = math.radians(sum(p[1] for p in ring) / len(ring))
    kx, ky = 111.320 * math.cos(lat0), 110.574
    a = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        a += (x1 * kx) * (y2 * ky) - (x2 * kx) * (y1 * ky)
    return abs(a) / 2.0


def polygon_area_km2(geom: dict) -> float:
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    total = 0.0
    for poly in polys:
        total += ring_area_km2(poly[0]) - sum(ring_area_km2(h) for h in poly[1:])
    return total


def bbox(geom: dict) -> tuple[float, float, float, float]:
    xs, ys = zip(*_coords(geom))
    return min(xs), min(ys), max(xs), max(ys)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(raw: Path, derived: Path, site: Path, manifest: Path) -> tuple[dict, list[str]]:
    problems: list[str] = []
    prev = json.loads(manifest.read_text()) if manifest.exists() else {}

    # -- boundary ---------------------------------------------------------
    b = json.loads((raw / "fan_boundary.geojson").read_text())
    if len(b["features"]) != 1:
        raise ContractError(f"boundary: expected 1 feature, got {len(b['features'])}")
    bgeom = b["features"][0]["geometry"]
    if bgeom["type"] not in ("Polygon", "MultiPolygon"):
        raise ContractError(f"boundary: geometry is {bgeom['type']}")
    bx = bbox(bgeom)
    if not (EUGENE[0] <= bx[0] and bx[2] <= EUGENE[2] and EUGENE[1] <= bx[1] and bx[3] <= EUGENE[3]):
        problems.append(f"boundary: bbox {bx} is outside Eugene (projection bug?)")
    area = polygon_area_km2(bgeom)
    if not BOUNDARY_KM2[0] <= area <= BOUNDARY_KM2[1]:
        problems.append(f"boundary: area {area:.2f} km2 outside {BOUNDARY_KM2}")

    # -- taxlots ----------------------------------------------------------
    t = json.loads((raw / "fan_taxlots.geojson").read_text())
    n = len(t["features"])
    if n < TAXLOTS_MIN:
        problems.append(f"taxlots: only {n} features (min {TAXLOTS_MIN})")
    last_n = (prev.get("taxlots") or {}).get("count")
    if last_n and abs(n - last_n) / last_n > TAXLOT_SWING:
        problems.append(f"taxlots: count moved {last_n} -> {n} (> {TAXLOT_SWING:.0%}); truncated feed?")
    pad = 0.01
    strays = 0
    for f in t["features"]:
        g = f.get("geometry")
        if not g:
            strays += 1
            continue
        x, y = next(_coords(g))
        if not (bx[0] - pad <= x <= bx[2] + pad and bx[1] - pad <= y <= bx[3] + pad):
            strays += 1
    if strays:
        problems.append(f"taxlots: {strays} features outside the FAN box or without geometry")
    ids = [(f.get("properties") or {}).get("maptaxlot") for f in t["features"]]
    have = [i for i in ids if i]
    if n and len(have) / n < TAXLOT_ID_COVERAGE:
        problems.append(f"taxlots: only {len(have)}/{n} have a maptaxlot id")

    result: dict = {
        "boundary": {"area_km2": round(area, 3), "sha256": sha256(raw / "fan_boundary.geojson")},
        "taxlots": {"count": n, "unique_ids": len(set(have)), "sha256": sha256(raw / "fan_taxlots.geojson")},
    }

    # -- heat (optional until the first grid exists) ----------------------
    meta_p = derived / "heat_scenes.json"
    layers_p = site / "layers.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if len(meta["scenes"]) < HEAT_MIN_SCENES:
            problems.append(f"heat: only {len(meta['scenes'])} scenes")
        med = meta.get("fan_median_f")
        if med is None or not HEAT_MEDIAN_F[0] <= med <= HEAT_MEDIAN_F[1]:
            problems.append(f"heat: FAN median {med} F is implausible (scaling bug?)")
        lots = json.loads((site / "taxlots.geojson").read_text())["features"] if (site / "taxlots.geojson").exists() else []
        deltas = [f["properties"].get("heat_delta_f") for f in lots]
        valued = [d for d in deltas if d is not None]
        if lots and len(valued) / len(lots) < HEAT_LOT_COVERAGE:
            problems.append(f"heat: only {len(valued)}/{len(lots)} taxlots got a value")
        if valued and max(abs(d) for d in valued) > HEAT_MAX_ABS_DELTA:
            problems.append(f"heat: a taxlot is {max(abs(d) for d in valued):.1f} F off the median")
        if not layers_p.exists() or "heat" not in json.loads(layers_p.read_text()):
            problems.append("heat: grid exists but the site was built without it")
        result["heat"] = {
            "summers": meta["summers"],
            "scenes": [s["id"] for s in meta["scenes"]],
            "fan_median_f": med,
            "sha256": sha256(derived / "heat_f.tif"),
        }

    # -- green cover (optional until the first grid exists) ----------------
    gmeta_p = derived / "green_scenes.json"
    if gmeta_p.exists():
        g = json.loads(gmeta_p.read_text())
        if len(g["scenes"]) < GREEN_MIN_SCENES:
            problems.append(f"green: only {len(g['scenes'])} scenes")
        med = g.get("fan_median_ndvi")
        if med is None or not GREEN_MEDIAN_NDVI[0] <= med <= GREEN_MEDIAN_NDVI[1]:
            problems.append(f"green: FAN median NDVI {med} is implausible (offset or band bug?)")
        lots = json.loads((site / "taxlots.geojson").read_text())["features"] if (site / "taxlots.geojson").exists() else []
        pcts = [f["properties"].get("green_pct") for f in lots]
        valued = [p for p in pcts if p is not None]
        if lots and len(valued) / len(lots) < GREEN_LOT_COVERAGE:
            problems.append(f"green: only {len(valued)}/{len(lots)} taxlots got a value")
        if any(not 0 <= p <= 100 for p in valued):
            problems.append("green: a taxlot share is outside 0-100%")
        if not layers_p.exists() or "green" not in json.loads(layers_p.read_text()):
            problems.append("green: grid exists but the site was built without it")
        result["green"] = {
            "summers": g["summers"],
            "scenes": [s["id"] for s in g["scenes"]],
            "fan_median_ndvi": med,
            "fan_green_share": g.get("fan_green_share"),
            "sha256": sha256(derived / "green_ndvi.tif"),
        }

    return result, problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gate the publish on sane data.")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--derived", type=Path, default=Path("data/derived"))
    ap.add_argument("--site", type=Path, default=Path("site/data"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    ap.add_argument("--write", action="store_true", help="record a passing run in the manifest")
    args = ap.parse_args(argv)

    try:
        result, problems = check(args.raw, args.derived, args.site, args.manifest)
    except (ContractError, FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        print(f"CONTRACT FAILED: {exc}", file=sys.stderr)
        return 1

    if problems:
        print("CONTRACT FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print(f"contract ok: {result['taxlots']['count']} taxlots, "
          f"boundary {result['boundary']['area_km2']} km2, heat={'heat' in result}, green={'green' in result}")
    if args.write:
        result["checked_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
