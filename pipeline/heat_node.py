#!/usr/bin/env python3
"""
Neighborhood Sensorium :: heat node (Landsat land surface temperature).

Builds one summer heat grid for the FAN neighborhood from Landsat 8/9
Collection 2 Level-2 surface temperature (band ST_B10, "lwir11"), read straight
from Microsoft Planetary Computer's public STAC catalog. No account, no key.

Method, in plain terms:
  1. Find every Landsat 8/9 scene over FAN from mid June to mid September of
     the last few complete summers, with low scene cloud cover.
  2. For each scene, read only the small window over FAN, reprojected onto one
     fixed 30 m grid (UTM 10N), and throw away cloudy, shadowed or missing
     pixels using the scene's own QA band.
  3. Keep a scene only if most of FAN is clear in it.
  4. Take the per-pixel median across the kept scenes. That is a "typical clear
     summer day" surface temperature, robust to one freak heat wave or one
     bad cloud mask.

Caveats that belong on the methods page:
  * Landsat passes over Eugene at about 11:15 AM Pacific, so this is late
    morning surface temperature, not the 4 PM peak and not air temperature.
  * The thermal sensor's native resolution is 100 m; USGS delivers it
    resampled to 30 m. Differences between neighbouring taxlots are not real.
    Differences between blocks and parts of the neighbourhood are.

The output (data/derived/heat_f.tif plus heat_scenes.json) is small and is
committed, so the weekly run only recomputes it when the target summers change.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import rasterio
import requests
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform as warp_xy
from rasterio.warp import transform_geom

LOG = logging.getLogger("sensorium.heat")

STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/landsat-c2-l2"
COLLECTION = "landsat-c2-l2"
USER_AGENT = "NeighborhoodSensorium/0.2 (civic GIS pipeline; github.com/fgc-gh/fan-sensorium)"

GRID_CRS = "EPSG:32610"  # UTM zone 10N, the native Landsat CRS over Eugene
PIXEL = 30.0             # metres
BUFFER_M = 300.0         # context around the boundary, so the edge isn't a cliff

SEASON = ((6, 15), (9, 15))  # month, day inclusive
MAX_SCENE_CLOUD = 20         # percent, whole-scene metadata
MIN_AOI_CLEAR = 0.80         # fraction of FAN pixels that must be clear
MIN_SCENES = 3

# Landsat C2 L2 surface temperature scaling (USGS product guide).
ST_SCALE, ST_OFFSET = 0.00341802, 149.0
# QA_PIXEL bits: 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow, 5 snow.
QA_BAD = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "GDAL_HTTP_USERAGENT": USER_AGENT,
}


# --------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    crs: str
    transform: rasterio.Affine
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


def grid_for(boundary_geom: dict, pixel: float = PIXEL, buffer_m: float = BUFFER_M) -> Grid:
    """A pixel-snapped UTM grid covering the boundary plus a buffer."""
    g = transform_geom("EPSG:4326", GRID_CRS, boundary_geom)
    xs, ys = zip(*_all_coords(g))
    minx = np.floor((min(xs) - buffer_m) / pixel) * pixel
    maxx = np.ceil((max(xs) + buffer_m) / pixel) * pixel
    miny = np.floor((min(ys) - buffer_m) / pixel) * pixel
    maxy = np.ceil((max(ys) + buffer_m) / pixel) * pixel
    width, height = int(round((maxx - minx) / pixel)), int(round((maxy - miny) / pixel))
    return Grid(GRID_CRS, from_origin(minx, maxy, pixel, pixel), width, height)


def _all_coords(geom: dict) -> Iterable[tuple[float, float]]:
    def walk(c):
        if isinstance(c[0], (int, float)):
            yield (c[0], c[1])
        else:
            for x in c:
                yield from walk(x)
    yield from walk(geom["coordinates"])


def aoi_mask(boundary_geom: dict, grid: Grid) -> np.ndarray:
    """True for pixels whose centre falls inside the boundary."""
    g = transform_geom("EPSG:4326", grid.crs, boundary_geom)
    return features.geometry_mask([g], grid.shape, grid.transform, invert=True)


def lonlat_bbox(grid: Grid) -> list[float]:
    t = grid.transform
    xs = [t.c, t.c + grid.width * t.a]
    ys = [t.f, t.f + grid.height * t.e]
    lon, lat = warp_xy(grid.crs, "EPSG:4326", [xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]])
    return [min(lon), min(lat), max(lon), max(lat)]


# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------


def target_summers(today: dt.date, n: int) -> list[int]:
    """The last `n` summers whose season window has fully closed."""
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
            "query": {
                "eo:cloud_cover": {"lt": MAX_SCENE_CLOUD},
                "platform": {"in": ["landsat-8", "landsat-9"]},
            },
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
        LOG.info("summer %d: %d candidate scenes so far", year, len(items))
    items.sort(key=lambda it: it["properties"]["datetime"])
    return items


def sas_signer(s: requests.Session) -> Callable[[str], str]:
    r = s.get(SAS_TOKEN, timeout=30)
    r.raise_for_status()
    token = r.json()["token"]
    return lambda href: f"{href}?{token}" if "?" not in href else href


# --------------------------------------------------------------------------
# Raster
# --------------------------------------------------------------------------


def read_on_grid(href: str, grid: Grid) -> np.ndarray:
    with rasterio.open(href) as src, WarpedVRT(
        src, crs=grid.crs, transform=grid.transform, width=grid.width, height=grid.height,
        resampling=Resampling.nearest, src_nodata=0, nodata=0,
    ) as vrt:
        return vrt.read(1)


def scene_to_f(st: np.ndarray, qa: np.ndarray) -> np.ndarray:
    """Scaled ST_B10 DN -> degrees Fahrenheit, NaN where cloudy or missing."""
    kelvin = st.astype("float64") * ST_SCALE + ST_OFFSET
    out = (kelvin - 273.15) * 9.0 / 5.0 + 32.0
    bad = (st == 0) | ((qa.astype("uint32") & QA_BAD) != 0)
    out[bad] = np.nan
    return out.astype("float32")


def composite(
    items: list[dict], grid: Grid, aoi: np.ndarray, sign: Callable[[str], str],
    reader: Callable[[str, Grid], np.ndarray] = read_on_grid,
) -> tuple[np.ndarray, list[dict]]:
    stack, used = [], []
    for it in items:
        try:
            st = reader(sign(it["assets"]["lwir11"]["href"]), grid)
            qa = reader(sign(it["assets"]["qa_pixel"]["href"]), grid)
        except Exception as exc:  # noqa: BLE001 -- one bad scene must not sink the run
            LOG.warning("%s: unreadable (%s); skipped", it["id"], exc)
            continue
        f = scene_to_f(st, qa)
        clear = float(np.isfinite(f[aoi]).mean()) if aoi.any() else 0.0
        if clear < MIN_AOI_CLEAR:
            LOG.info("%s: %.0f%% of FAN clear; skipped", it["id"], clear * 100)
            continue
        stack.append(f)
        used.append({
            "id": it["id"],
            "datetime": it["properties"]["datetime"],
            "platform": it["properties"].get("platform"),
            "fan_clear": round(clear, 3),
            "fan_median_f": round(float(np.nanmedian(f[aoi])), 1),
        })
        LOG.info("%s: kept (FAN median %.1f F)", it["id"], used[-1]["fan_median_f"])
    # WRS rows overlap: one pass over Eugene often arrives as two scenes (row
    # 029 and row 030) with the same date. Count each pass once, keeping the
    # copy that saw more of FAN, so no single day gets double weight.
    best: dict[str, int] = {}
    for i, u in enumerate(used):
        day = u["datetime"][:10]
        if day not in best or u["fan_clear"] > used[best[day]]["fan_clear"]:
            best[day] = i
    keep = sorted(best.values())
    if len(keep) < len(used):
        LOG.info("%d scenes are duplicate passes; %d distinct days kept", len(used) - len(keep), len(keep))
    stack, used = [stack[i] for i in keep], [used[i] for i in keep]
    if not stack:
        return np.full(grid.shape, np.nan, dtype="float32"), used
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns stay NaN
        comp = np.nanmedian(np.stack(stack), axis=0).astype("float32")
    return comp, used


def write_grid(arr: np.ndarray, grid: Grid, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=grid.width, height=grid.height, count=1,
        dtype="float32", crs=grid.crs, transform=grid.transform, nodata=np.nan,
        compress="deflate", predictor=3,
    ) as dst:
        dst.write(arr, 1)
        dst.update_tags(units="degF", description="median clear-sky summer land surface temperature")


# --------------------------------------------------------------------------


def run(
    boundary_path: Path, outdir: Path, summers: list[int], force: bool = False,
    s: requests.Session | None = None,
    search: Callable[..., list[dict]] = search_scenes,
    signer: Callable[[requests.Session], Callable[[str], str]] = sas_signer,
    reader: Callable[[str, Grid], np.ndarray] = read_on_grid,
) -> int:
    meta_path, tif_path = outdir / "heat_scenes.json", outdir / "heat_f.tif"
    if not force and meta_path.exists() and tif_path.exists():
        prev = json.loads(meta_path.read_text())
        if prev.get("summers") == summers:
            LOG.info("heat grid already built for summers %s; nothing to do", summers)
            return 0

    boundary = json.loads(boundary_path.read_text())["features"][0]["geometry"]
    grid = grid_for(boundary)
    aoi = aoi_mask(boundary, grid)
    LOG.info("grid %dx%d px at %.0f m, %d px inside FAN", grid.width, grid.height, PIXEL, int(aoi.sum()))

    s = s or requests.Session()
    s.headers.setdefault("User-Agent", USER_AGENT)
    items = search(lonlat_bbox(grid), summers, s)
    with rasterio.Env(**GDAL_ENV):
        comp, used = composite(items, grid, aoi, signer(s), reader)

    if len(used) < MIN_SCENES:
        LOG.error("only %d usable scenes (need %d); not writing a heat grid", len(used), MIN_SCENES)
        return 1

    write_grid(comp, grid, tif_path)
    meta = {
        "summers": summers,
        "season": f"{SEASON[0][0]:02d}-{SEASON[0][1]:02d} to {SEASON[1][0]:02d}-{SEASON[1][1]:02d}",
        "source": "Landsat 8/9 Collection 2 Level-2 ST_B10 via Microsoft Planetary Computer",
        "statistic": "per-pixel median of clear scenes",
        "overpass_local": "about 11:15 AM Pacific",
        "fan_median_f": round(float(np.nanmedian(comp[aoi])), 2),
        "scenes": used,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    LOG.info("heat grid written from %d scenes; FAN median %.1f F", len(used), meta["fan_median_f"])
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the FAN summer surface-temperature grid.")
    ap.add_argument("--boundary", type=Path, default=Path("data/raw/fan_boundary.geojson"))
    ap.add_argument("--outdir", type=Path, default=Path("data/derived"))
    ap.add_argument("--summers", type=int, default=3, help="how many recent complete summers")
    ap.add_argument("--force", action="store_true", help="rebuild even if the summers are unchanged")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-7s %(name)s %(message)s", stream=sys.stderr)
    summers = target_summers(dt.date.today(), args.summers)
    return run(args.boundary, args.outdir, summers, force=args.force or bool(os.environ.get("HEAT_FORCE")))


if __name__ == "__main__":
    raise SystemExit(main())
