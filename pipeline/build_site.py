#!/usr/bin/env python3
"""
Neighborhood Sensorium :: site builder.

Joins the layers into what the web map loads, and nothing more:

  site/data/boundary.geojson   the FAN outline
  site/data/taxlots.geojson    taxlots, lean properties, plus each lot's heat
  site/data/heat.png           the heat grid as a coloured image overlay
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
ALPHA_IN, ALPHA_OUT = 205, 70  # inside vs outside the FAN line

TAXLOT_KEEP = ("maptaxlot", "mapacres")


def ramp_rgb(delta: np.ndarray) -> np.ndarray:
    stops = np.array([s for s, _ in RAMP])
    cols = np.array([c for _, c in RAMP], dtype="float64")
    d = np.clip(delta, stops[0], stops[-1])
    out = np.stack([np.interp(d, stops, cols[:, i]) for i in range(3)], axis=-1)
    return out.astype("uint8")


def zonal_means(geoms_wgs84: list[dict], grid: np.ndarray, transform, crs) -> list[float | None]:
    """Mean grid value under each polygon (all_touched, so tiny lots still get
    a value). None where the lot falls entirely outside the grid or on NaN."""
    h, w = grid.shape
    out: list[float | None] = []
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
        m = features.geometry_mask([gp], (r1 - r0, c1 - c0), win_t, invert=True, all_touched=True)
        vals = grid[r0:r1, c0:c1][m]
        vals = vals[np.isfinite(vals)]
        out.append(float(vals.mean()) if vals.size else None)
    return out


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

    heat_tif, heat_meta = derived / "heat_f.tif", derived / "heat_scenes.json"
    lot_heat: list[float | None] = [None] * len(taxlots["features"])
    if heat_tif.exists() and heat_meta.exists():
        meta = json.loads(heat_meta.read_text())
        with rasterio.open(heat_tif) as src:
            grid = src.read(1)
            transform, crs = src.transform, src.crs.to_string()
        aoi = features.geometry_mask([transform_geom("EPSG:4326", crs, bgeom)], grid.shape, transform, invert=True)
        median = float(np.nanmedian(grid[aoi]))
        delta = grid - median

        rgba = np.zeros(grid.shape + (4,), dtype="uint8")
        rgba[..., :3] = ramp_rgb(np.nan_to_num(delta))
        rgba[..., 3] = np.where(aoi, ALPHA_IN, ALPHA_OUT)
        rgba[~np.isfinite(grid), 3] = 0
        write_png(rgba, out / "heat.png")

        h, w = grid.shape
        px = [0, w, w, 0]
        py = [0, 0, h, h]
        xs, ys = zip(*(_xy(transform, c, r) for c, r in zip(px, py)))
        lon, lat = warp_xy(crs, "EPSG:4326", list(xs), list(ys))
        corners = [[round(a, 6), round(b, 6)] for a, b in zip(lon, lat)]  # TL, TR, BR, BL

        means = zonal_means([f.get("geometry") for f in taxlots["features"]], grid, transform, crs)
        lot_heat = [None if m is None else m - median for m in means]

        layers["heat"] = {
            "image": "data/heat.png",
            "corners": corners,
            "fan_median_f": round(median, 1),
            "summers": meta["summers"],
            "season": meta["season"],
            "overpass_local": meta["overpass_local"],
            "scene_count": len(meta["scenes"]),
            "first_scene": meta["scenes"][0]["datetime"][:10],
            "last_scene": meta["scenes"][-1]["datetime"][:10],
            "source": meta["source"],
            "ramp": [[s, "#%02x%02x%02x" % c] for s, c in RAMP],
            "lots_with_value": sum(v is not None for v in lot_heat),
        }
    else:
        LOG.warning("no heat grid in %s; publishing without the heat layer", derived)

    lean = []
    for f, d in zip(taxlots["features"], lot_heat):
        props = {k: (f.get("properties") or {}).get(k) for k in TAXLOT_KEEP}
        if d is not None:
            props["heat_delta_f"] = round(d, 1)
        lean.append({"type": "Feature", "geometry": _round_geom(f.get("geometry")), "properties": props})

    (out / "boundary.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": _round_geom(bgeom), "properties": {"name": "Friendly Area Neighbors"}}]},
        separators=(",", ":")))
    (out / "taxlots.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": lean}, separators=(",", ":")))
    (out / "layers.json").write_text(json.dumps(layers, indent=2) + "\n")
    LOG.info("site data written: %d taxlots, heat=%s", len(lean), "heat" in layers)
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
