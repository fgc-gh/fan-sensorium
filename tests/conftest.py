"""Synthetic FAN: a small square near the real neighbourhood, a grid of lots,
and fake Landsat scenes on disk. Everything offline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_xy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

LON0, LON1, LAT0, LAT1 = -123.145, -123.125, 44.025, 44.040


def square(x0, y0, x1, y1):
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


@pytest.fixture
def boundary_fc():
    return {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": square(LON0, LAT0, LON1, LAT1),
                      "properties": {"NAME": "Friendly Area Neighbors", "NEIBORHD": "FA"}}],
        "properties": {"source": "fake://boundary", "retrieved_utc": "2026-11-24T00:00:00Z"},
    }


@pytest.fixture
def taxlots_fc():
    feats, n = [], 24
    dx, dy = (LON1 - LON0) / n, (LAT1 - LAT0) / n
    for i in range(n):
        for j in range(n):
            x0, y0 = LON0 + i * dx, LAT0 + j * dy
            feats.append({"type": "Feature",
                          "geometry": square(x0 + dx * .05, y0 + dy * .05, x0 + dx * .95, y0 + dy * .95),
                          "properties": {"maptaxlot": f"1803{i:03d}{j:03d}", "mapacres": 0.4, "OBJECTID": i * n + j}})
    return {"type": "FeatureCollection", "features": feats,
            "properties": {"source": "fake://taxlots", "retrieved_utc": "2026-11-24T00:00:00Z"}}


@pytest.fixture
def raw_dir(tmp_path, boundary_fc, taxlots_fc):
    d = tmp_path / "raw"
    d.mkdir()
    (d / "fan_boundary.geojson").write_text(json.dumps(boundary_fc))
    (d / "fan_taxlots.geojson").write_text(json.dumps(taxlots_fc))
    return d


def f_to_dn(f):
    k = (np.asarray(f, dtype="float64") - 32) * 5 / 9 + 273.15
    return np.round((k - 149.0) / 0.00341802).astype("uint16")


@pytest.fixture
def scenes(tmp_path):
    """Five fake scenes in UTM 10N over the AOI.

    Truth: 85 F at the west edge rising to 97 F at the east edge.
      s0, s1, s2: truth + (-2, 0, +2) F        -> median == truth
      s3: truth + 30 F but half cloud          -> must be dropped (< 80% clear)
      s4: truth + 1 F with a cloud bit set in a corner outside FAN -> kept
      s5: same date as s0 (overlapping row), +40 F, sees less -> dropped as duplicate
    """
    xs, ys = warp_xy("EPSG:4326", "EPSG:32610", [LON0 - .01, LON1 + .01], [LAT1 + .01, LAT0 - .01])
    x0, y1 = xs[0], ys[0]
    w = int((xs[1] - xs[0]) // 30) + 1
    h = int((ys[0] - ys[1]) // 30) + 1
    t = from_origin(x0, y1, 30, 30)
    truth = np.tile(np.linspace(85, 97, w), (h, 1))

    def write(path, arr):
        with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1, dtype="uint16",
                           crs="EPSG:32610", transform=t, nodata=0) as dst:
            dst.write(arr.astype("uint16"), 1)

    clear_qa = np.full((h, w), 1 << 6, dtype="uint16")  # bit 6 = clear
    items = []
    specs = [(-2, None), (0, None), (2, None), (30, "half"), (1, "corner"), (40, "dupe")]
    for k, (off, cloud) in enumerate(specs):
        st = f_to_dn(truth + off)
        qa = clear_qa.copy()
        if cloud == "half":
            qa[:, : w // 2] = 1 << 3
        elif cloud == "corner":
            qa[:3, :3] = 1 << 3
        elif cloud == "dupe":  # same pass as s0 (overlapping WRS row), sees less of FAN
            qa[:, w // 2: w // 2 + 4] = 1 << 3  # a thin cloud strip across FAN
        sp, qp = tmp_path / f"s{k}_st.tif", tmp_path / f"s{k}_qa.tif"
        write(sp, st)
        write(qp, qa)
        items.append({"id": f"LC09_FAKE_{k}",
                      "properties": {"datetime": f"2025-07-{10 + (0 if cloud == 'dupe' else k):02d}T18:50:{k:02d}Z",
                                     "platform": "landsat-9"},
                      "assets": {"lwir11": {"href": str(sp)}, "qa_pixel": {"href": str(qp)}}})
    return items
