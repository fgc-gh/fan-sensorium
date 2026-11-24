from __future__ import annotations

import copy
import json

import numpy as np
import pytest

import build_site
import contract
import fetch_node
import green_node
import heat_node


# ---------------------------------------------------------------- fetch -----

class FakeArcGIS:
    """Answers describe/query calls the way the two real servers do."""

    def __init__(self, boundary_fc, taxlots_fc):
        self.b, self.t, self.calls = boundary_fc, taxlots_fc, []

    def __call__(self, s, url, params, retries=4, timeout=90):
        self.calls.append((url, dict(params)))
        if url.endswith("MapServer/1"):
            return {"name": "Neighborhoods", "maxRecordCount": 1000, "supportedQueryFormats": "JSON, geoJSON, PBF",
                    "advancedQueryCapabilities": {"supportsPagination": False}, "fields": [{"name": "FID", "type": "esriFieldTypeOID"}]}
        if url.endswith("FeatureServer/0"):
            return {"name": "Eugene Taxlots - HUB", "maxRecordCount": 100, "objectIdField": "OBJECTID",
                    "supportedQueryFormats": "JSON, geoJSON, PBF", "advancedQueryCapabilities": {"supportsPagination": True}}
        if "MapServer/1/query" in url:
            if params.get("returnIdsOnly") == "true":
                hit = params["where"] == "NAME = 'Friendly Area Neighbors'"
                return {"objectIds": [7] if hit else []}
            assert "resultOffset" not in params, "MapServer ignores resultOffset"
            return {"type": "FeatureCollection", "features": self.b["features"]}
        if "FeatureServer/0/query" in url:
            assert "geometry" in params, "taxlots must be clipped server-side"
            off, n = params["resultOffset"], params["resultRecordCount"]
            page = self.t["features"][off: off + n]
            more = off + n < len(self.t["features"])
            return {"type": "FeatureCollection", "features": page, "properties": {"exceededTransferLimit": more}}
        raise AssertionError(url)


def test_fetch_pages_both_server_styles(tmp_path, monkeypatch, boundary_fc, taxlots_fc):
    fake = FakeArcGIS(boundary_fc, taxlots_fc)
    monkeypatch.setattr(fetch_node, "_post", fake)
    assert fetch_node.run(tmp_path) == 0
    lots = json.loads((tmp_path / "fan_taxlots.geojson").read_text())
    assert len(lots["features"]) == len(taxlots_fc["features"])  # 576 lots over 6 pages of 100
    assert sum("FeatureServer/0/query" in u for u, _ in fake.calls) == 6


def test_fetch_falls_back_when_name_changes(tmp_path, monkeypatch, boundary_fc, taxlots_fc):
    fake = FakeArcGIS(boundary_fc, taxlots_fc)
    orig = fake.__call__

    def renamed(s, url, params, **kw):
        if params.get("returnIdsOnly") == "true" and "LIKE" in params.get("where", ""):
            return {"objectIds": [7]}
        if params.get("returnIdsOnly") == "true":
            return {"objectIds": []}
        return orig(s, url, params, **kw)

    monkeypatch.setattr(fetch_node, "_post", renamed)
    assert fetch_node.run(tmp_path, only=["boundary"]) == 0
    b = json.loads((tmp_path / "fan_boundary.geojson").read_text())
    assert "LIKE" in b["properties"]["where"]


def test_write_if_changed_ignores_timestamp(tmp_path, boundary_fc):
    p = tmp_path / "b.geojson"
    assert fetch_node.write_if_changed(boundary_fc, p) is True
    again = copy.deepcopy(boundary_fc)
    again["properties"]["retrieved_utc"] = "2030-01-01T00:00:00Z"
    assert fetch_node.write_if_changed(again, p) is False
    again["features"][0]["properties"]["NAME"] = "x"
    assert fetch_node.write_if_changed(again, p) is True


# ----------------------------------------------------------------- heat -----

def run_heat(raw_dir, out, scenes, summers=(2023, 2024, 2025), force=False):
    return heat_node.run(raw_dir / "fan_boundary.geojson", out, list(summers), force=force,
                         search=lambda bbox, summers, s: scenes, signer=lambda s: (lambda h: h))


def test_heat_composite_masks_clouds_and_takes_median(tmp_path, raw_dir, scenes):
    out = tmp_path / "derived"
    assert run_heat(raw_dir, out, scenes) == 0
    meta = json.loads((out / "heat_scenes.json").read_text())
    kept = [s["id"] for s in meta["scenes"]]
    assert "LC09_FAKE_3" not in kept, "half-cloudy +30 F scene must be dropped"
    assert "LC09_FAKE_5" not in kept and "LC09_FAKE_0" in kept, "duplicate pass must count once"
    assert len(kept) == 4
    # Truth runs 85..97 F across the grid; median of (-2, 0, +2, +1) offsets is +0.5.
    assert 88 < meta["fan_median_f"] < 95

    import rasterio
    with rasterio.open(out / "heat_f.tif") as src:
        g = src.read(1)
    west, east = np.nanmean(g[:, :5]), np.nanmean(g[:, -5:])
    assert east - west > 5, "west-to-east gradient must survive"


def test_heat_skips_when_summers_unchanged(tmp_path, raw_dir, scenes):
    out = tmp_path / "derived"
    assert run_heat(raw_dir, out, scenes) == 0
    calls = []
    heat_node.run(raw_dir / "fan_boundary.geojson", out, [2023, 2024, 2025],
                  search=lambda *a: calls.append(1) or scenes, signer=lambda s: (lambda h: h))
    assert calls == []


def test_heat_refuses_too_few_scenes(tmp_path, raw_dir, scenes):
    assert run_heat(raw_dir, tmp_path / "d", scenes[:2]) == 1
    assert not (tmp_path / "d" / "heat_f.tif").exists()


def test_scene_to_f_scaling():
    dn = np.array([[0, 44177]], dtype="uint16")  # 0 = fill; 44177 ~ 300 K
    qa = np.zeros_like(dn)
    f = heat_node.scene_to_f(dn, qa)
    assert np.isnan(f[0, 0])
    assert abs(f[0, 1] - 80.33) < 0.1


def test_target_summers():
    import datetime as dt
    assert heat_node.target_summers(dt.date(2026, 11, 24), 3) == [2024, 2025, 2026]
    assert heat_node.target_summers(dt.date(2026, 8, 1), 3) == [2023, 2024, 2025]


# ---------------------------------------------------------------- green -----

def run_green(raw_dir, out, scenes, summers=(2026,), force=False):
    return green_node.run(raw_dir / "fan_boundary.geojson", out, list(summers), force=force,
                          search=lambda bbox, summers, s: scenes, signer=lambda s: (lambda h: h))


def test_reflectance_offset_by_baseline():
    dn = np.array([3000], dtype="uint16")
    assert green_node.reflectance(dn, "05.11")[0] == pytest.approx(0.2)
    assert green_node.reflectance(dn, "03.01")[0] == pytest.approx(0.3)


def test_scene_ndvi_masks_cloud_and_nodata():
    red = np.array([[1800, 1800, 0]], dtype="uint16")
    nir = np.array([[4200, 4200, 4200]], dtype="uint16")
    scl = np.array([[4, 9, 4]], dtype="uint16")
    ndvi = green_node.scene_ndvi(red, nir, scl, "05.11")
    assert ndvi[0, 0] == pytest.approx((0.32 - 0.08) / 0.40)
    assert np.isnan(ndvi[0, 1]) and np.isnan(ndvi[0, 2])


def test_green_composite(tmp_path, raw_dir, s2_scenes):
    out = tmp_path / "derived"
    assert run_green(raw_dir, out, s2_scenes) == 0
    meta = json.loads((out / "green_scenes.json").read_text())
    ids = [s["id"] for s in meta["scenes"]]
    assert ids == ["S2_FAKE_0", "S2_FAKE_1", "S2_FAKE_2", "S2_FAKE_3"]  # cloud dropped, duplicate day dropped
    import rasterio
    with rasterio.open(out / "green_ndvi.tif") as src:
        g = src.read(1)
        assert src.res == (10.0, 10.0)
    row = g[g.shape[0] // 2]
    row = row[np.isfinite(row)]
    assert row[0] > 0.6 and row[-1] < 0.3  # leafy west, paved east
    assert 0.3 < meta["fan_median_ndvi"] < 0.6
    assert 0.3 < meta["fan_green_share"] < 0.7


def test_green_refuses_too_few_scenes(tmp_path, raw_dir, s2_scenes):
    assert run_green(raw_dir, tmp_path / "d", s2_scenes[:2]) == 1
    assert not (tmp_path / "d" / "green_ndvi.tif").exists()


# ------------------------------------------------------- site + contract ----

def build_all(tmp_path, raw_dir, scenes):
    derived, site = tmp_path / "derived", tmp_path / "site"
    assert run_heat(raw_dir, derived, scenes) == 0
    layers = build_site.build(raw_dir, derived, site)
    return derived, site, layers


def test_site_build_and_contract_pass(tmp_path, raw_dir, scenes):
    derived, site, layers = build_all(tmp_path, raw_dir, scenes)
    assert (site / "heat.png").exists() and not (site / "heat.png.aux.xml").exists()
    assert len(layers["heat"]["corners"]) == 4
    lots = json.loads((site / "taxlots.geojson").read_text())["features"]
    assert set(lots[0]["properties"]) >= {"maptaxlot", "mapacres", "heat_delta_f"}
    # West lots cooler than east lots.
    west = [f["properties"]["heat_delta_f"] for f in lots if f["properties"]["maptaxlot"].startswith("1803000")]
    east = [f["properties"]["heat_delta_f"] for f in lots if f["properties"]["maptaxlot"].startswith("1803023")]
    assert max(west) < 0 < min(east)

    manifest = tmp_path / "manifest.json"
    assert contract.main(["--raw", str(raw_dir), "--derived", str(derived), "--site", str(site),
                          "--manifest", str(manifest), "--write"]) == 0
    m = json.loads(manifest.read_text())
    assert m["taxlots"]["count"] == 576 and len(m["heat"]["scenes"]) == 4


def test_contract_blocks_truncated_feed(tmp_path, raw_dir, scenes, taxlots_fc):
    derived, site, _ = build_all(tmp_path, raw_dir, scenes)
    manifest = tmp_path / "manifest.json"
    args = ["--raw", str(raw_dir), "--derived", str(derived), "--site", str(site), "--manifest", str(manifest)]
    assert contract.main(args + ["--write"]) == 0
    cut = dict(taxlots_fc, features=taxlots_fc["features"][:340])  # a 40% drop
    (raw_dir / "fan_taxlots.geojson").write_text(json.dumps(cut))
    assert contract.main(args) == 1


def test_contract_blocks_projection_bug(tmp_path, raw_dir, scenes, taxlots_fc):
    derived, site, _ = build_all(tmp_path, raw_dir, scenes)
    bad = copy.deepcopy(taxlots_fc)
    # Coordinates in Oregon State Plane feet, served as if they were lon/lat.
    bad["features"][0]["geometry"]["coordinates"] = [[[4240000, 870000], [4240100, 870000], [4240100, 870100], [4240000, 870000]]]
    (raw_dir / "fan_taxlots.geojson").write_text(json.dumps(bad))
    ok, problems = contract.check(raw_dir, derived, site, tmp_path / "m.json")
    assert any("outside the FAN box" in p for p in problems)


def test_site_with_green_tells_the_story(tmp_path, raw_dir, scenes, s2_scenes):
    derived, site = tmp_path / "derived", tmp_path / "site"
    assert run_heat(raw_dir, derived, scenes) == 0
    assert run_green(raw_dir, derived, s2_scenes) == 0
    layers = build_site.build(raw_dir, derived, site)
    assert (site / "green.png").exists() and len(layers["green"]["corners"]) == 4
    lots = json.loads((site / "taxlots.geojson").read_text())["features"]
    west = [f["properties"]["green_pct"] for f in lots if f["properties"]["maptaxlot"].startswith("1803000")]
    east = [f["properties"]["green_pct"] for f in lots if f["properties"]["maptaxlot"].startswith("1803023")]
    assert min(west) == 100 and max(east) == 0
    story = layers["story"]
    assert story["gap_f"] > 4 and story["lots"] == 576  # bare lots run hotter (truth: ~4.8 F)
    manifest = tmp_path / "manifest.json"
    assert contract.main(["--raw", str(raw_dir), "--derived", str(derived), "--site", str(site),
                          "--manifest", str(manifest), "--write"]) == 0
    assert len(json.loads(manifest.read_text())["green"]["scenes"]) == 4


def test_contract_blocks_green_offset_bug(tmp_path, raw_dir, scenes, s2_scenes):
    derived, site = tmp_path / "derived", tmp_path / "site"
    run_heat(raw_dir, derived, scenes)
    run_green(raw_dir, derived, s2_scenes)
    build_site.build(raw_dir, derived, site)
    meta = json.loads((derived / "green_scenes.json").read_text())
    meta["fan_median_ndvi"] = -0.4
    (derived / "green_scenes.json").write_text(json.dumps(meta))
    problems = contract.check(raw_dir, derived, site, tmp_path / "m.json")[1]
    assert any("NDVI" in p for p in problems)


def test_site_builds_without_heat(tmp_path, raw_dir):
    layers = build_site.build(raw_dir, tmp_path / "nothing", tmp_path / "site")
    assert "heat" not in layers
    assert contract.check(raw_dir, tmp_path / "nothing", tmp_path / "site", tmp_path / "m.json")[1] == []
