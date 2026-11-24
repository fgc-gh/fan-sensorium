#!/usr/bin/env python3
"""
Neighborhood Sensorium :: fetch node (ArcGIS REST ingest).

Fetches public ArcGIS layers, filters them server-side against the Friendly
Area Neighbors (FAN) polygon, and writes RFC 7946 GeoJSON to data/raw/.

Design constraints:
  * requests + stdlib only. Runs in a plain GitHub Actions runner in seconds.
  * All filtering is pushed to the server. We never download the county.
  * Content-addressed output: if a payload is unchanged, the file is not
    rewritten, so the weekly job produces no noise commit.

Usage:
    python pipeline/fetch_node.py
    python pipeline/fetch_node.py --only boundary
    python pipeline/fetch_node.py --outdir data/raw --verbose

Fertile Ground Conservancy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import requests

LOG = logging.getLogger("sensorium.fetch")

WGS84 = 4326
USER_AGENT = "NeighborhoodSensorium/0.2 (civic GIS pipeline; github.com/fgc-gh/fan-sensorium)"

CITY = "https://gis.eugene-or.gov/arcgis/rest/services"
LCOG = "https://services3.arcgis.com/F7NiRLGNbA2hh7gE/arcgis/rest/services"


# --------------------------------------------------------------------------
# Registry. Every endpoint the pipeline touches is declared here, once.
# Adding a vector layer to the Sensorium should be a five-line diff.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Layer:
    key: str
    url: str
    out_fields: str = "*"
    where: str = "1=1"
    # Tried in order if `where` returns nothing (e.g. the city renames a row).
    fallback_where: tuple[str, ...] = ()
    note: str = ""


LAYERS: dict[str, Layer] = {
    "boundary": Layer(
        key="boundary",
        url=f"{CITY}/Common/EugeneBoundaries/MapServer/1",
        out_fields="NAME,NEIBORHD",
        where="NAME = 'Friendly Area Neighbors'",
        fallback_where=("UPPER(NAME) LIKE '%FRIENDLY%'",),
        note="City of Eugene neighborhood association boundaries (MapServer, no pagination).",
    ),
    "taxlots": Layer(
        key="taxlots",
        url=f"{LCOG}/EugTaxlots/FeatureServer/0",
        out_fields="maptaxlot,mapacres",
        note="LCOG / Lane County A&T taxlots, Eugene extract. The join spine.",
    ),
}

AOI_KEY = "boundary"


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _post(s: requests.Session, url: str, params: dict[str, Any], retries: int = 4, timeout: int = 90) -> dict:
    """POST, not GET: geometry rings and objectId lists exceed the query-string
    limit some Esri front ends enforce, and that failure looks like a dead layer."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = s.post(url, data=params, timeout=timeout)
            r.raise_for_status()
            payload = r.json()
            # ArcGIS returns HTTP 200 with an error envelope. Check explicitly.
            if isinstance(payload, dict) and "error" in payload:
                raise RuntimeError(f"ArcGIS error {payload['error']}")
            return payload
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt == retries - 1:
                break
            backoff = 2**attempt
            LOG.warning("%s -> %s; retrying in %ss", url, exc, backoff)
            time.sleep(backoff)
    raise RuntimeError(f"exhausted retries for {url}") from last


# --------------------------------------------------------------------------
# Layer introspection
# --------------------------------------------------------------------------


@dataclass
class LayerCapabilities:
    max_record_count: int
    supports_pagination: bool
    supports_geojson: bool
    oid_field: str
    name: str
    raw: dict = field(repr=False, default_factory=dict)


def describe(s: requests.Session, layer: Layer) -> LayerCapabilities:
    meta = _post(s, layer.url, {"f": "json"})
    adv = meta.get("advancedQueryCapabilities") or {}
    formats = (meta.get("supportedQueryFormats") or "").lower()

    oid = meta.get("objectIdField") or ""
    if not oid:
        for fld in meta.get("fields") or []:
            if fld.get("type") == "esriFieldTypeOID":
                oid = fld["name"]
                break
    caps = LayerCapabilities(
        max_record_count=int(meta.get("maxRecordCount") or 1000),
        supports_pagination=bool(adv.get("supportsPagination")),
        supports_geojson="geojson" in formats,
        oid_field=oid or "OBJECTID",
        name=meta.get("name", layer.key),
        raw=meta,
    )
    LOG.info("%s: %r max=%d pagination=%s geojson=%s", layer.key, caps.name,
             caps.max_record_count, caps.supports_pagination, caps.supports_geojson)
    if not caps.supports_geojson:
        raise RuntimeError(f"{layer.key}: server will not emit GeoJSON")
    return caps


# --------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------


def esri_polygon(geom: dict) -> dict:
    """GeoJSON Polygon/MultiPolygon -> Esri polygon (flat ring list).

    Winding is not reordered: for a spatial *filter* every server tested
    accepts either, and a wrong guess on a hole is worse than a slightly
    larger candidate set (the contract catches anything that strays)."""
    t, coords = geom["type"], geom["coordinates"]
    if t == "Polygon":
        rings = coords
    elif t == "MultiPolygon":
        rings = [ring for poly in coords for ring in poly]
    else:
        raise ValueError(f"cannot use {t} as an AOI filter")
    return {"rings": rings, "spatialReference": {"wkid": WGS84}}


def _base_params(layer: Layer, aoi: dict | None) -> dict[str, Any]:
    p: dict[str, Any] = {
        "f": "geojson",
        "outFields": layer.out_fields,
        "outSR": WGS84,
        "returnGeometry": "true",
        # Esri "true curves" are silently dropped by naive GeoJSON consumers.
        "returnTrueCurves": "false",
    }
    if aoi is not None:
        p.update({
            "geometry": json.dumps(esri_polygon(aoi)),
            "geometryType": "esriGeometryPolygon",
            "spatialRel": "esriSpatialRelIntersects",
            "inSR": WGS84,
        })
    return p


def _page_by_offset(s, layer: Layer, caps: LayerCapabilities, params: dict, where: str) -> Iterator[dict]:
    """Hosted FeatureServers (services*.arcgis.com) support resultOffset."""
    offset, page = 0, caps.max_record_count
    while True:
        q = dict(params, where=where, resultOffset=offset, resultRecordCount=page,
                 orderByFields=caps.oid_field)
        fc = _post(s, f"{layer.url}/query", q)
        feats = fc.get("features") or []
        yield from feats
        exceeded = (fc.get("properties") or {}).get("exceededTransferLimit") or fc.get("exceededTransferLimit")
        if not feats or (not exceeded and len(feats) < page):
            return
        offset += len(feats)


def _page_by_oid(s, layer: Layer, caps: LayerCapabilities, params: dict, where: str) -> Iterator[dict]:
    """Legacy MapServers (gis.eugene-or.gov) report supportsPagination=false and
    silently ignore resultOffset, so a naive loop re-downloads page one forever.
    Enumerate object ids, then fetch in windows."""
    ids_q = dict(params, where=where, returnIdsOnly="true", f="json")
    ids_q.pop("outFields", None)
    resp = _post(s, f"{layer.url}/query", ids_q)
    oids = resp.get("objectIds") or (resp.get("properties") or {}).get("objectIds") or []
    oids = sorted(oids)
    LOG.info("%s: %d object ids", layer.key, len(oids))
    step = caps.max_record_count
    for i in range(0, len(oids), step):
        q = dict(params, objectIds=",".join(str(o) for o in oids[i:i + step]))
        fc = _post(s, f"{layer.url}/query", q)
        yield from fc.get("features") or []


def fetch(s: requests.Session, layer: Layer, aoi: dict | None = None) -> dict:
    """Return a GeoJSON FeatureCollection for `layer`, clipped to `aoi`."""
    caps = describe(s, layer)
    params = _base_params(layer, aoi)
    pager = _page_by_offset if caps.supports_pagination else _page_by_oid

    used_where = layer.where
    features: list[dict] = []
    for where in (layer.where, *layer.fallback_where):
        features = list(pager(s, layer, caps, params, where))
        used_where = where
        if features:
            break
        LOG.warning("%s: no features for where=%r", layer.key, where)

    LOG.info("%s: %d features", layer.key, len(features))
    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "source": layer.url,
            "layer": caps.name,
            "where": used_where,
            "clipped_to": AOI_KEY if aoi is not None else None,
            "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }


# --------------------------------------------------------------------------
# Sink
# --------------------------------------------------------------------------


def content_digest(fc: dict) -> str:
    """sha256 of the payload, ignoring the retrieval timestamp."""
    body = dict(fc)
    props = dict(body.get("properties") or {})
    props.pop("retrieved_utc", None)
    body["properties"] = props
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_if_changed(fc: dict, path: Path) -> bool:
    """Write only on content change. Returns True if the file was rewritten."""
    digest = content_digest(fc)
    stamp = path.with_suffix(path.suffix + ".sha256")
    if path.exists() and stamp.exists() and stamp.read_text().strip() == digest:
        LOG.info("%s: unchanged (%s)", path.name, digest[:12])
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fc, separators=(",", ":")))
    stamp.write_text(digest + "\n")
    LOG.info("%s: wrote %d features, %.1f KB", path.name, len(fc["features"]), path.stat().st_size / 1024)
    return True


# --------------------------------------------------------------------------


def run(outdir: Path, only: list[str] | None = None, s: requests.Session | None = None) -> int:
    s = s or session()
    wanted = only or list(LAYERS)

    # 1. The AOI is fetched first and unconditionally; everything is relative to it.
    aoi_fc = fetch(s, LAYERS[AOI_KEY], aoi=None)
    if len(aoi_fc["features"]) != 1:
        LOG.error("boundary query returned %d features, expected exactly 1. Check the NAME "
                  "literal against the layer's values before assuming the server is down.",
                  len(aoi_fc["features"]))
        return 1
    aoi_geom = aoi_fc["features"][0]["geometry"]
    write_if_changed(aoi_fc, outdir / "fan_boundary.geojson")

    # 2. Everything else is clipped to it, server-side.
    for key in wanted:
        if key == AOI_KEY:
            continue
        fc = fetch(s, LAYERS[key], aoi=aoi_geom)
        write_if_changed(fc, outdir / f"fan_{key}.geojson")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch FAN vector layers from public ArcGIS services.")
    ap.add_argument("--outdir", type=Path, default=Path("data/raw"))
    ap.add_argument("--only", choices=sorted(LAYERS), action="append")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-7s %(name)s %(message)s", stream=sys.stderr)
    return run(args.outdir, args.only)


if __name__ == "__main__":
    raise SystemExit(main())
