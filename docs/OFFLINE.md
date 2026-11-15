# Testing with downloaded data (offline / IRSA outages)

IRSA's image search, cutout service and Gaia/CatWISE cone search are sometimes
unavailable (HTTP 502 "Proxy Error"). The app can then run its **unchanged**
pipeline on data that were downloaded earlier: real SPHEREx cutouts and real
catalogue rows, nothing synthetic. Code: `backend/app/localdata.py`.

## Run it

```bash
python scripts/serve.py --mode cached     # offline: downloaded data only, no network at all
python scripts/serve.py                   # auto (default): IRSA first, downloaded data if IRSA fails
python scripts/serve.py --mode live       # IRSA only (no fallback)
```

Then open http://127.0.0.1:8000/. Options: `--port`, `--host`, `--data DIR`.
With plain uvicorn, set the environment variable instead:
`SPHEREX_DATA_MODE=cached uvicorn app.main:app` (from `backend/`;
`SPHEREX_LOCAL_DATA=DIR` points at another data directory).

| mode | images | catalogues | object names |
|---|---|---|---|
| `live` | IRSA | IRSA | SIMBAD / Sesame |
| `auto` (default) | IRSA; the local copy if the IRSA search fails and the position is a downloaded field | IRSA; the local rows if the query fails (or straight away when the images are local) | SIMBAD; the saved SIMBAD result if resolution fails |
| `cached` | local copy only | local rows only | saved results of downloaded fields, or typed coordinates; never the network |

## Downloaded fields

The bundled fields (`backend/tests/data/real/`) are the five regression targets:

| query | detectors | field | exposures |
|---|---|---|---|
| Barnard's star | D3, D6 | 5′ | 36 |
| eps Ind B | D3, D6 | 5′ | 47 |
| RW Aur | D3, D6 | 5′ | 36 |
| WISE J085510.83-071442.5 | D5, D6 | 5′ | 33 |
| R Dra | D3, D6 | 5′ | 29 |

Search by these names (any spacing or capitalisation, e.g. "barnards star") or
by coordinates within 2′ of a field. `GET /api/local-data` lists the mode and
the fields.

A position that was never downloaded gives a clear error that lists what is
available. More fields can be added with `scripts/cache_validation_cutouts.py`
(cutouts) and `scripts/cache_catalogs.py` (Gaia/CatWISE rows) while IRSA is up.

## What is different from a live search

- **Detectors:** only the detectors that were downloaded. Others are reported as "Not downloaded".
- **Field of view:** at most the downloaded field. A larger request is reduced, and the banner says so.
- **Exposures:** capped per detector and epoch as at download time, so S/N can be lower than live.
- **Catalogues:** only rows within the cached cone (0.07°).
- **Click-to-spectrum:** uses the same local exposures.

## How it is shown

- **Banner:** a "Local data" banner above the viewer gives the reason (offline mode, or the IRSA error) and the field's download date.
- **Frames:** every frame carries `data_source: "local_cache"`, and its label says "local copy downloaded earlier (not live IRSA)".
- **Analysis:** `data_source` and `catalog_source` (per catalogue: `irsa_live` / `local_cache`) record where each part came from. Any fallback is listed among the catalogue notes.
- **Browser cache:** local results are **not** stored in it, so once IRSA is back the same search queries it again.
