#!/usr/bin/env python3
"""
Neighborhood Sensorium :: green-cover node (Sentinel-2 summer NDVI).

First pass at the canopy layer. Builds a 10 m summer greenness grid (NDVI)
for FAN from Sentinel-2 Level-2A imagery on Microsoft Planetary Computer.
No account, no key.

Method, in plain terms:
  1. Find every Sentinel-2 pass over FAN from mid June to mid September of
     the latest complete summer, with low scene cloud cover.
  2. Read only the window over FAN (red, near-infrared and the scene
     classification band), reprojected onto one fixed 10 m UTM grid.
  3. Drop cloud, shadow, cirrus and missing pixels using the classification
     band. Keep a pass only if most of FAN is clear; count each date once.
  4. NDVI = (NIR - red) / (NIR + red) per pass; per-pixel median across passes.

"Green cover" is the share of ground with summer NDVI >= 0.5.

Read this honestly:
  * NDVI sees anything leafy and growing. In a dry Eugene summer that is
    mostly trees, but irrigated lawns and gardens count too. This is a
    green-cover proxy, not a tree-canopy measurement. Phase 3b replaces it
    with lidar canopy height, which separates a tree from a lawn.
  * 10 m pixels: compare blocks, not individual small lots.

Fertile Ground Conservancy.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
import requests

from heat_node import (GDAL_ENV, SEASON, USER_AGENT, Grid, aoi_mask, grid_for,
                       lonlat_bbox, read_on_grid)

LOG = logging.getLogger("sensorium.green")

STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
COLLECTION = "sentinel-2-l2a"
SAS_TOKEN = f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{COLLECTION}"

PIXEL = 10.0
MAX_SCENE_CLOUD = 20
MIN_AOI_CLEAR = 0.80
MIN_SCENES = 3
GREEN_NDVI = 0.5

# Scene classification (SCL) values that are usable ground.
SCL_OK = {2, 4, 5, 6, 7}  # dark area, vegetation, bare soil, water, unclassified
# Since processing baseline 04.00 (Jan 2022), L2A reflectance carries a +1000 offset.
BOA_OFFSET_FROM = "04.00"


def target_summers(today: dt.date, n: int) -> list[int]:
    last = today.year if today >= dt.date(today.year, *SEASON[1]) + dt.timedelta(days=30) else today.year - 1
    return list(range(last - n + 1, last + 1))


def search_scenes(bbox: list[float], summers: list[int], s: requests.Session) -> list[dict]:
    items: list[dict] = []
    for year in summers:
        (m0, d0), (m1, d1) = SEASON
        body = {
            "collections": [COLLECTION],
            "bbox": bbox,
            "datetime": f"{year}-{m0:02d}-{d0:02d}T00:00:00Z/{year}-{m1:02d}-{d1:02d}T23:59:59Z",
            "query": {"eo:cloud_cover": {"lt": MAX_SCENE_CLOUD}},
            "limit": 100,
        }
        page = s.post(STAC_SEARCH, json=body, timeout=60)
        while True:
            page.raise_for_status()
            data = page.json()
            items.extend(data.get("features", []))
            nxt = next((ln for ln in data.get("links", []) if ln.get("rel") == "next"), None)
            if not nxt:
                break
            if nxt.get("method", "GET").upper() == "POST":
                page = s.post(nxt["href"], json=nxt.get("body", body), timeout=60)
            else:
                page = s.get(nxt["href"], timeout=60)
        LOG.info("summer %d: %d candidate passes so far", year, len(items))
    items.sort(key=lambda it: it["properties"]["datetime"])
    return items


def sas_signer(s: requests.Session) -> Callable[[str], str]:
    r = s.get(SAS_TOKEN, timeout=30)
    r.raise_for_status()
    token = r.json()["token"]
    return lambda href: f"{href}?{token}" if "?" not in href else href


def reflectance(dn: np.ndarray, baseline: str | None) -> np.ndarray:
    r = dn.astype("float32")
    if baseline and baseline >= BOA_OFFSET_FROM:
        r = r - 1000.0
    return r / 10000.0


def scene_ndvi(red_dn: np.ndarray, nir_dn: np.ndarray, scl: np.ndarray, baseline: str | None) -> np.ndarray:
    red, nir = reflectance(red_dn, baseline), reflectance(nir_dn, baseline)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red)
    bad = ~np.isin(scl, list(SCL_OK)) | (red_dn == 0) | (nir_dn == 0) | ((nir + red) <= 0)
    ndvi[bad] = np.nan
    return np.clip(ndvi, -1, 1).astype("float32")


def composite(items: list[dict], grid: Grid, aoi: np.ndarray, sign: Callable[[str], str],
              reader: Callable[[str, Grid], np.ndarray] = read_on_grid) -> tuple[np.ndarray, list[dict]]:
    by_day: dict[str, tuple[np.ndarray, dict]] = {}
    for it in items:
        try:
            red = reader(sign(it["assets"]["B04"]["href"]), grid)
            nir = reader(sign(it["assets"]["B08"]["href"]), grid)
            scl = reader(sign(it["assets"]["SCL"]["href"]), grid)
        except Exception as exc:  # noqa: BLE001 -- one bad pass must not sink the run
            LOG.warning("%s: unreadable (%s); skipped", it["id"], exc)
            continue
        ndvi = scene_ndvi(red, nir, scl, it["properties"].get("s2:processing_baseline"))
        clear = float(np.isfinite(ndvi[aoi]).mean()) if aoi.any() else 0.0
        if clear < MIN_AOI_CLEAR:
            LOG.info("%s: %.0f%% of FAN clear; skipped", it["id"], clear * 100)
            continue
        day = it["properties"]["datetime"][:10]
        info = {"id": it["id"], "datetime": it["properties"]["datetime"], "fan_clear": round(clear, 3),
                "fan_median_ndvi": round(float(np.nanmedian(ndvi[aoi])), 3)}
        # Neighbouring MGRS tiles deliver one pass twice. Count each date once.
        if day not in by_day or clear > by_day[day][1]["fan_clear"]:
            by_day[day] = (ndvi, info)
        LOG.info("%s: kept (FAN median NDVI %.2f)", it["id"], info["fan_median_ndvi"])
    days = sorted(by_day)
    if not days:
        return np.full(grid.shape, np.nan, dtype="float32"), []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        comp = np.nanmedian(np.stack([by_day[d][0] for d in days]), axis=0).astype("float32")
    return comp, [by_day[d][1] for d in days]


def write_grid(arr: np.ndarray, grid: Grid, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=grid.width, height=grid.height, count=1,
                       dtype="float32", crs=grid.crs, transform=grid.transform, nodata=np.nan,
                       compress="deflate", predictor=3) as dst:
        dst.write(arr, 1)
        dst.update_tags(description="median clear-sky summer NDVI, Sentinel-2 L2A")


def run(boundary_path: Path, outdir: Path, summers: list[int], force: bool = False,
        s: requests.Session | None = None,
        search: Callable[..., list[dict]] = search_scenes,
        signer: Callable[[requests.Session], Callable[[str], str]] = sas_signer,
        reader: Callable[[str, Grid], np.ndarray] = read_on_grid) -> int:
    meta_path, tif_path = outdir / "green_scenes.json", outdir / "green_ndvi.tif"
    if not force and meta_path.exists() and tif_path.exists():
        if json.loads(meta_path.read_text()).get("summers") == summers:
            LOG.info("green grid already built for summers %s; nothing to do", summers)
            return 0

    boundary = json.loads(boundary_path.read_text())["features"][0]["geometry"]
    grid = grid_for(boundary, pixel=PIXEL)
    aoi = aoi_mask(boundary, grid)
    LOG.info("grid %dx%d px at %.0f m, %d px inside FAN", grid.width, grid.height, PIXEL, int(aoi.sum()))

    s = s or requests.Session()
    s.headers.setdefault("User-Agent", USER_AGENT)
    items = search(lonlat_bbox(grid), summers, s)
    with rasterio.Env(**GDAL_ENV):
        comp, used = composite(items, grid, aoi, signer(s), reader)
    if len(used) < MIN_SCENES:
        LOG.error("only %d usable passes (need %d); not writing a green grid", len(used), MIN_SCENES)
        return 1

    write_grid(comp, grid, tif_path)
    inside = comp[aoi]
    inside = inside[np.isfinite(inside)]
    meta = {
        "summers": summers,
        "season": f"{SEASON[0][0]:02d}-{SEASON[0][1]:02d} to {SEASON[1][0]:02d}-{SEASON[1][1]:02d}",
        "source": "Sentinel-2 Level-2A (ESA Copernicus) via Microsoft Planetary Computer",
        "statistic": "per-pixel median NDVI of clear passes",
        "green_threshold_ndvi": GREEN_NDVI,
        "fan_median_ndvi": round(float(np.median(inside)), 3),
        "fan_green_share": round(float((inside >= GREEN_NDVI).mean()), 3),
        "scenes": used,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    LOG.info("green grid written from %d passes; %.0f%% of FAN is green (NDVI >= %.1f)",
             len(used), meta["fan_green_share"] * 100, GREEN_NDVI)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the FAN summer green-cover (NDVI) grid.")
    ap.add_argument("--boundary", type=Path, default=Path("data/raw/fan_boundary.geojson"))
    ap.add_argument("--outdir", type=Path, default=Path("data/derived"))
    ap.add_argument("--summers", type=int, default=1, help="how many recent complete summers")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-7s %(name)s %(message)s", stream=sys.stderr)
    summers = target_summers(dt.date.today(), args.summers)
    return run(args.boundary, args.outdir, summers, force=args.force or bool(os.environ.get("GREEN_FORCE")))


if __name__ == "__main__":
    raise SystemExit(main())
