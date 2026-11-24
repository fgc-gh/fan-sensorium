# Neighborhood Sensorium

A free public map of the Friendly Area Neighbors (FAN) neighborhood in Eugene,
Oregon, that shows where the ground runs hottest on a clear summer day, built
entirely from public data and refreshed automatically. Built and hosted by
Fertile Ground Conservancy as a gift to FAN.

> "It's a map that shows which blocks in our neighborhood run hottest, so we
> can bring real numbers to the city and put our energy where it'll help most."

## What's here now (Phase 1 Spine + Phase 2 Heat)

| Piece | File | What it does |
|---|---|---|
| Fetch | `pipeline/fetch_node.py` | Pulls the FAN boundary (City of Eugene) and every taxlot inside it (Lane County via LCOG), filtered server-side. |
| Heat | `pipeline/heat_node.py` | Median clear-sky summer surface temperature on a 30 m grid, from every usable Landsat 8/9 pass over the last three summers (Planetary Computer, no key). |
| Site data | `pipeline/build_site.py` | Heat image overlay, each taxlot's "°F above or below the FAN typical", lean GeoJSON for the map. |
| Contract | `pipeline/contract.py` | Refuses to publish if the boundary changes shape, a feed truncates, coordinates come back in the wrong projection, or the heat numbers are implausible. |
| Map | `site/index.html` | MapLibre map, mobile-friendly, tap a lot for its number. |
| Automation | `.github/workflows/refresh.yml` | Weekly run on Mondays. Tests, fetches, checks, publishes to GitHub Pages. |

Nothing needs a password, an API key or a server. The heat grid is
recomputed only when a new summer completes; it is small and committed under
`data/derived/` so the weekly run stays fast.

### What the heat number means, honestly

- Landsat passes over Eugene at about 11:15 AM, so this is late-morning
  **ground** temperature on clear days, not afternoon air temperature.
- The thermal sensor sees about 100 m at a time (delivered at 30 m). Compare
  blocks and parts of the neighborhood, not next-door lots.
- It's relative: each lot is shown against the FAN-wide median, which is the
  number that survives a hostile read ("this block runs 5 °F hotter than the
  rest of the neighborhood").

## Run it on your laptop

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q tests              # offline, synthetic data
python pipeline/fetch_node.py          # boundary + taxlots -> data/raw/
python pipeline/heat_node.py           # Landsat -> data/derived/ (a few minutes the first time)
python pipeline/build_site.py          # -> site/data/
python pipeline/contract.py            # must say "contract ok"
python -m http.server -d site 8000     # open http://localhost:8000
```

## Go live (one-time GitHub settings)

1. **Settings → Pages → Build and deployment → Source:** GitHub Actions.
2. **Settings → Actions → General → Workflow permissions:** Read and write.
3. Merge to `main`. The workflow runs, commits the heat grid and manifest,
   and publishes to `https://fgc-gh.github.io/fan-sensorium/`.

Every push to any branch also runs the whole pipeline against live data and
attaches the built site as a downloadable `site-preview` artifact, without
publishing anything.

## When the weekly run goes red

That's the system working: the contract blocked a bad week and last week's
map is still up. Open the failed run in the Actions tab; the "Check contract"
step says what moved. Common causes: the city renamed or split the
neighborhood polygon, or a county feed came back truncated. If the change is
real (say, a big lot split), rerun with the limit loosened in `contract.py`.

## Roadmap

1. ~~**Spine.** Boundary, taxlots, pipeline, live map.~~
2. ~~**Heat.** Landsat surface temperature.~~
3. **Canopy.** Sentinel-2 NDVI proxy first, then lidar canopy height. City street
   trees as a separate, labeled overlay (it lists only city-owned trees).
4. **Water.** Impervious surface, slope, stormwater network proximity.
5. **Activation.** Printable one-page brief per block, micro-grant target list,
   methods page.

## License

Not chosen yet. Until a license file is added, the default is all rights
reserved.
