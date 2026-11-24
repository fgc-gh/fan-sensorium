#!/usr/bin/env python3
"""
Neighborhood Sensorium :: site builder.

Joins the layers into what the web map loads, and nothing more:

  site/data/boundary.geojson   the FAN outline
  site/data/taxlots.geojson    taxlots, lean properties, plus each lot's heat
  site/data/heat.png           the heat grid as a coloured image overlay
  site/data/green.png          the summer green-cover grid, same treatment
  site/data/layers.json        corners for the image, legend, sources, dates

Heat per taxlot is the mean of the grid pixels the lot touches, reported as
degrees F above or below the FAN-wide median. That relative number is the one
a councilor can repeat: "this block runs 6 degrees hotter than the
neighbourhood typical".

Fertile Ground Conservancy.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import rasterio
import rasterio.errors
from rasterio import features
from rasterio.warp import transform as warp_xy
from rasterio.warp import transform_geom

LOG = logging.getLogger("sensorium.site")

# Delta (degrees F from FAN median) -> colour. Earth tones: cool sage through
# sand to rust and oxblood. Kept in one place; the legend in the page reads it.
RAMP = [
    (-6.0, (93, 122, 110)),   # sage
    (-3.0, (150, 164, 132)),  # lichen
    (0.0, (222, 205, 160)),   # sand
    (3.0, (205, 133, 63)),    # ochre
    (6.0, (166, 61, 36)),     # rust
    (9.0, (102, 24, 24)),     # oxblood
]
# Summer NDVI -> colour: dry straw and pavement through moss to deep fir.
GREEN_RAMP = [
    (0.0, (196, 182, 150)),   # pavement / roof
    (0.25, (214, 196, 140)),  # dry grass
    (0.45, (170, 172, 110)),  # sparse green
    (0.6, (112, 138, 84)),    # moss
    (0.8, (52, 82, 47)),      # fir
]
ALPHA_IN, ALPHA_OUT = 205, 70  # inside vs outside the FAN line

TAXLOT_KEEP = ("maptaxlot", "mapacres")


def ramp_rgb(values: np.ndarray, ramp=RAMP) -> np.ndarray:
    stops = np.array([v for v, _ in ramp])
    cols = np.array([c for _, c in ramp], dtype="float64")
    d = np.clip(values, stops[0], stops[-1])
    out = np.stack([np.interp(d, stops, cols[:, i]) for i in range(3)], axis=-1)
    return out.astype("uint8")


def zonal_values(geoms_wgs84: list[dict], grid: np.ndarray, transform, crs,
                 all_touched: bool = True) -> list[np.ndarray | None]:
    """Finite grid values under each polygon. With all_touched=False, a lot too
    small to contain any pixel centre falls back to every pixel it touches, so
    every lot inside the grid gets a reading. None where the lot is off-grid."""
    h, w = grid.shape
    out: list[np.ndarray | None] = []
    for g in geoms_wgs84:
        if not g:
            out.append(None)
            continue
        gp = transform_geom("EPSG:4326", crs, g)
        # Rasterize only within the lot's own pixel window: thousands of lots stay fast.
        xs, ys = zip(*_coords(gp))
        cols, rows = zip(*(_rowcol(transform, x, y) for x, y in ((min(xs), max(ys)), (max(xs), min(ys)))))
        c0, c1 = max(int(np.floor(min(cols))) - 1, 0), min(int(np.ceil(max(cols))) + 1, w)
        r0, r1 = max(int(np.floor(min(rows))) - 1, 0), min(int(np.ceil(max(rows))) + 1, h)
        if c0 >= c1 or r0 >= r1:
            out.append(None)
            continue
        win_t = rasterio.Affine(transform.a, transform.b, transform.c + c0 * transform.a + r0 * transform.b,
                                transform.d, transform.e, transform.f + c0 * transform.d + r0 * transform.e)
        win = grid[r0:r1, c0:c1]
        vals = np.array([])
        for touched in ((True,) if all_touched else (False, True)):
            m = features.geometry_mask([gp], win.shape, win_t, invert=True, all_touched=touched)
            vals = win[m]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                break
        out.append(vals if vals.size else None)
    return out


def zonal_means(geoms_wgs84: list[dict], grid: np.ndarray, transform, crs) -> list[float | None]:
    return [None if v is None else float(v.mean())
            for v in zonal_values(geoms_wgs84, grid, transform, crs, all_touched=True)]


def _read_grid(tif: Path, bgeom: dict):
    with rasterio.open(tif) as src:
        grid = src.read(1)
        transform, crs = src.transform, src.crs.to_string()
    aoi = features.geometry_mask([transform_geom("EPSG:4326", crs, bgeom)], grid.shape, transform, invert=True)
    return grid, transform, crs, aoi


def _corners(grid: np.ndarray, transform, crs) -> list[list[float]]:
    h, w = grid.shape
    xs, ys = zip(*(_xy(transform, c, r) for c, r in zip([0, w, w, 0], [0, 0, h, h])))
    lon, lat = warp_xy(crs, "EPSG:4326", list(xs), list(ys))
    return [[round(a, 6), round(b, 6)] for a, b in zip(lon, lat)]  # TL, TR, BR, BL


def _overlay(values: np.ndarray, valid: np.ndarray, aoi: np.ndarray, ramp, path: Path) -> None:
    rgba = np.zeros(values.shape + (4,), dtype="uint8")
    rgba[..., :3] = ramp_rgb(np.nan_to_num(values), ramp)
    rgba[..., 3] = np.where(aoi, ALPHA_IN, ALPHA_OUT)
    rgba[~valid, 3] = 0
    write_png(rgba, path)


def _xy(t, col: float, row: float) -> tuple[float, float]:
    """Pixel corner -> map coordinates (spelled out; affine's operator API is in flux)."""
    return t.c + col * t.a + row * t.b, t.f + col * t.d + row * t.e


def _rowcol(t, x: float, y: float) -> tuple[float, float]:
    """Map coordinates -> fractional (col, row) for a north-up grid."""
    return (x - t.c) / t.a, (y - t.f) / t.e


def _coords(geom: dict):
    def walk(c):
        if isinstance(c[0], (int, float)):
            yield (c[0], c[1])
        else:
            for x in c:
                yield from walk(x)
    yield from walk(geom["coordinates"])


def write_png(rgba: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = rgba.shape
    warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
    with warnings.catch_warnings(), rasterio.open(path, "w", driver="PNG", width=w, height=h, count=4, dtype="uint8") as dst:
        for i in range(4):
            dst.write(rgba[:, :, i], i + 1)
    aux = path.with_suffix(path.suffix + ".aux.xml")
    if aux.exists():
        aux.unlink()


def build(raw: Path, derived: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    boundary = json.loads((raw / "fan_boundary.geojson").read_text())
    taxlots = json.loads((raw / "fan_taxlots.geojson").read_text())
    bgeom = boundary["features"][0]["geometry"]

    layers: dict = {
        "boundary": {"source": boundary["properties"]["source"], "retrieved_utc": boundary["properties"].get("retrieved_utc")},
        "taxlots": {"source": taxlots["properties"]["source"], "retrieved_utc": taxlots["properties"].get("retrieved_utc"),
                    "count": len(taxlots["features"])},
    }

    geoms = [f.get("geometry") for f in taxlots["features"]]
    n = len(geoms)

    # -- heat -----------------------------------------------------------------
    heat_tif, heat_meta = derived / "heat_f.tif", derived / "heat_scenes.json"
    lot_heat: list[float | None] = [None] * n
    if heat_tif.exists() and heat_meta.exists():
        meta = json.loads(heat_meta.read_text())
        grid, transform, crs, aoi = _read_grid(heat_tif, bgeom)
        median = float(np.nanmedian(grid[aoi]))
        _overlay(grid - median, np.isfinite(grid), aoi, RAMP, out / "heat.png")
        lot_heat = [None if m is None else m - median for m in zonal_means(geoms, grid, transform, crs)]

        vals = np.array([d for d in lot_heat if d is not None])
        if vals.size:
            p5, p50, p95 = np.percentile(vals, [5, 50, 95])
            LOG.info("lot heat vs FAN median: min %+.1f, p5 %+.1f, p50 %+.1f, p95 %+.1f, max %+.1f F",
                     vals.min(), p5, p50, p95, vals.max())
        layers["heat"] = {
            "image": "data/heat.png",
            "corners": _corners(grid, transform, crs),
            "fan_median_f": round(median, 1),
            "summers": meta["summers"],
            "season": meta["season"],
            "overpass_local": meta["overpass_local"],
            "scene_count": len(meta["scenes"]),
            "first_scene": meta["scenes"][0]["datetime"][:10],
            "last_scene": meta["scenes"][-1]["datetime"][:10],
            "source": meta["source"],
            "ramp": [[v, "#%02x%02x%02x" % c] for v, c in RAMP],
            "lots_with_value": sum(v is not None for v in lot_heat),
        }
    else:
        LOG.warning("no heat grid in %s; publishing without the heat layer", derived)

    # -- green cover ----------------------------------------------------------
    green_tif, green_meta = derived / "green_ndvi.tif", derived / "green_scenes.json"
    lot_green: list[float | None] = [None] * n
    lot_ndvi: list[float | None] = [None] * n
    if green_tif.exists() and green_meta.exists():
        meta = json.loads(green_meta.read_text())
        thr = meta["green_threshold_ndvi"]
        grid, transform, crs, aoi = _read_grid(green_tif, bgeom)
        _overlay(grid, np.isfinite(grid), aoi, GREEN_RAMP, out / "green.png")
        inside = grid[aoi]
        inside = inside[np.isfinite(inside)]
        per_lot = zonal_values(geoms, grid, transform, crs, all_touched=False)
        lot_green = [None if v is None else float((v >= thr).mean()) for v in per_lot]
        # Mean NDVI ranks lots for the story; the share alone ties at 0% and 100%.
        lot_ndvi = [None if v is None else float(v.mean()) for v in per_lot]
        layers["green"] = {
            "image": "data/green.png",
            "corners": _corners(grid, transform, crs),
            "threshold_ndvi": thr,
            "fan_green_share": round(float((inside >= thr).mean()), 3),
            "summers": meta["summers"],
            "season": meta["season"],
            "scene_count": len(meta["scenes"]),
            "first_scene": meta["scenes"][0]["datetime"][:10],
            "last_scene": meta["scenes"][-1]["datetime"][:10],
            "source": meta["source"],
            "ramp": [[v, "#%02x%02x%02x" % c] for v, c in GREEN_RAMP],
            "lots_with_value": sum(v is not None for v in lot_green),
        }
        LOG.info("green cover: %.0f%% of FAN, %d lots with a value",
                 layers["green"]["fan_green_share"] * 100, layers["green"]["lots_with_value"])
    else:
        LOG.warning("no green grid in %s; publishing without the green layer", derived)

    # -- the one sentence: how much hotter are the least green lots? -----------
    trios = [(v, g, h) for v, g, h in zip(lot_ndvi, lot_green, lot_heat) if None not in (v, g, h)]
    if len(trios) >= 50:
        trios.sort(key=lambda t: t[0])
        k = len(trios) // 5
        pairs = [(g, h) for _, g, h in trios]
        bare = float(np.mean([h for _, h in pairs[:k]]))
        leafy = float(np.mean([h for _, h in pairs[-k:]]))
        layers["story"] = {
            "least_green_fifth_heat_f": round(bare, 1),
            "greenest_fifth_heat_f": round(leafy, 1),
            "gap_f": round(bare - leafy, 1),
            "least_green_fifth_green_share": round(float(np.mean([g for g, _ in pairs[:k]])), 2),
            "greenest_fifth_green_share": round(float(np.mean([g for g, _ in pairs[-k:]])), 2),
            "lots": len(pairs),
        }
        LOG.info("least-green fifth of lots runs %.1f F hotter than the greenest fifth", bare - leafy)

    lean = []
    for f, d, g in zip(taxlots["features"], lot_heat, lot_green):
        props = {k: (f.get("properties") or {}).get(k) for k in TAXLOT_KEEP}
        if d is not None:
            props["heat_delta_f"] = round(d, 1)
        if g is not None:
            props["green_pct"] = int(round(g * 100))
        lean.append({"type": "Feature", "geometry": _round_geom(f.get("geometry")), "properties": props})

    (out / "boundary.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": _round_geom(bgeom), "properties": {"name": "Friendly Area Neighbors"}}]},
        separators=(",", ":")))
    (out / "taxlots.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": lean}, separators=(",", ":")))
    (out / "layers.json").write_text(json.dumps(layers, indent=2) + "\n")
    LOG.info("site data written: %d taxlots, heat=%s, green=%s", len(lean), "heat" in layers, "green" in layers)
    return layers


def _round_geom(g: dict | None, nd: int = 6) -> dict | None:
    """~10 cm precision; cuts the taxlot file roughly in half."""
    if not g:
        return g

    def r(c):
        return [round(c[0], nd), round(c[1], nd)] if isinstance(c[0], (int, float)) else [r(x) for x in c]
    return {"type": g["type"], "coordinates": r(g["coordinates"])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the web map's data files.")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--derived", type=Path, default=Path("data/derived"))
    ap.add_argument("--out", type=Path, default=Path("site/data"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s %(message)s", stream=sys.stderr)
    build(args.raw, args.derived, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
