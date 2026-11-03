"""
Puffin data setup -- GERMANY (Zensus 100m population grid).

Builds data/processed/<region>/cells.parquet: 100m fine cells with 64-dim
AlphaEarth features + Zensus labels. Works for a single city or the whole country,
in either of two modes:

  ALL_CELLS = False  -> only cells that carry a Zensus label   -> <REGION>/
  ALL_CELLS = True   -> every 100m land cell, pop=0 on empty land -> <REGION>_all/

Steps (each is skipped if its output already exists):
  1. Download Zensus 2022 CSVs from Destatis      (SKIP_ZENSUS)
  2. Download AlphaEarth (AEF) tiles for the bbox (SKIP_AEF)
  3. Build cells.parquet

Edit the CONFIG block below, then run:
    python setup_germany.py

For US voting data use setup_us_voting.py instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

from puffin.config import AE_RAW_DIR, DATA_PROCESSED, ZENSUS_RAW_DIR

# =============================================================================
# CONFIG -- edit these values, then run: python setup_germany.py
# =============================================================================
REGION = "germany"   # which area to build. Keys come from REGION_BBOX in
                     # src/puffin/data.py:
                     #   cities : berlin | leipzig | munich | dresden
                     #            nuremberg | hannover
                     #   country: germany
                     # To add a new city, add its EPSG:3035 bbox to REGION_BBOX.

YEAR   = 2022        # Zensus/AEF year
FORCE  = False       # re-download / rebuild even if outputs already exist

ALL_CELLS = True     # False -> labelled cells only            -> data/processed/<REGION>/
                     # True  -> every land cell incl. pop=0    -> data/processed/<REGION>_all/

BOUNDARY = "data/raw/boundaries/germany.geojson"
                     # Border polygon, used only when ALL_CELLS=True. Cells outside
                     # the border are DROPPED (not labelled 0). Required for a
                     # border-crossing bbox like `germany`, otherwise foreign towns
                     # would be treated as empty German land. Set None for a city
                     # bbox that lies safely inside the country.

SKIP_ZENSUS = True   # True = Zensus CSVs already in data/raw/zensus/
SKIP_AEF    = True   # True = AEF tiles already in data/raw/aef/<YEAR>/
# =============================================================================


# -- Step 1: Zensus CSVs ------------------------------------------------------
# To add a new target variable, add its URL and expected CSV filename below.
# URL pattern: https://www.destatis.de/static/DE/zensus/gitterdaten/<filename>.zip

ZENSUS_URLS = {
    "population":      "https://www.destatis.de/static/DE/zensus/gitterdaten/Zensus2022_Bevoelkerungszahl.zip",
    "citizens_18plus": "https://www.destatis.de/static/DE/zensus/gitterdaten/Zensus2022_Bevoelkerungszahl.zip",
    "rent_per_m2":     "https://www.destatis.de/static/DE/zensus/gitterdaten/Zensus2022_Durchschn_Nettokaltmiete.zip",
    "avg_age":         "https://www.destatis.de/static/DE/zensus/gitterdaten/Durchschnittsalter_in_Gitterzellen.zip",
    "vacancy_rate":    "https://www.destatis.de/static/DE/zensus/gitterdaten/Zensus2022_Leerstandsquote.zip",
    "foreigner_share": "https://www.destatis.de/static/DE/zensus/gitterdaten/Auslaenderanteil_in_Gitterzellen.zip",
}

# Expected CSV file relative to ZENSUS_RAW_DIR, used to skip if already present
ZENSUS_CHECK_FILES = {
    "population":      "population/Zensus2022_Bevoelkerungszahl_100m-Gitter.csv",
    "citizens_18plus": "population/Zensus2022_Bevoelkerungszahl_100m-Gitter.csv",
    "rent_per_m2":     "rent/Zensus2022_Durchschn_Nettokaltmiete_100m-Gitter.csv",
    "avg_age":         "age/Zensus2022_Durchschnittsalter_100m-Gitter.csv",
    "vacancy_rate":    "vacancy/Zensus2022_Leerstandsquote_100m-Gitter.csv",
    "foreigner_share": "foreigner_share/Zensus2022_Anteil_Auslaender_100m-Gitter.csv",
}


def download_zensus(force: bool = False) -> None:
    import urllib.request
    import zipfile
    import io

    ZENSUS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for var, url in ZENSUS_URLS.items():
        check = ZENSUS_RAW_DIR / ZENSUS_CHECK_FILES[var]
        if check.exists() and not force:
            print(f"  [zensus] {var}: already downloaded")
            continue
        print(f"  [zensus] Downloading {var} ...", end=" ", flush=True)
        with urllib.request.urlopen(url) as r:
            data = r.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(ZENSUS_RAW_DIR)
        print("done")


# -- Step 2: AEF tiles --------------------------------------------------------
# To support a new country: add its bbox to REGION_BBOX in data.py and ensure
# aef_index.parquet covers the needed UTM zones.

def download_aef(region: str, year: int = 2022, force: bool = False) -> None:
    """Download AEF tiles covering the region bbox."""
    import pandas as pd
    from pyproj import Transformer
    from puffin.data import REGION_BBOX

    bbox = REGION_BBOX.get(region)
    if bbox is None:
        print(f"  [aef] Unknown region '{region}' -- skipping tile download")
        return

    index_path = AE_RAW_DIR / "aef_index.parquet"
    if not index_path.exists():
        print("  [aef] ERROR: aef_index.parquet not found in data/raw/aef/")
        print("        Copy it from the archive or download from AlphaEarth.")
        return

    idx = pd.read_parquet(index_path)
    t = Transformer.from_crs(3035, 4326, always_xy=True)
    lon_w, lat_s = t.transform(bbox[0], bbox[1])
    lon_e, lat_n = t.transform(bbox[2], bbox[3])

    hits = idx[
        (idx.year == year) &
        (idx.utm_zone.isin(["31N", "32N", "33N"])) &
        (idx.wgs84_east  >= lon_w) & (idx.wgs84_west  <= lon_e) &
        (idx.wgs84_north >= lat_s) & (idx.wgs84_south <= lat_n)
    ]
    print(f"  [aef] {len(hits)} tiles needed for region={region}")

    import urllib.request
    downloaded = already = 0
    for _, row in hits.iterrows():
        zone  = row["utm_zone"]
        fname = Path(row["path"]).name
        dest  = AE_RAW_DIR / str(year) / zone / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not force:
            already += 1
            continue
        url = row.get("url") or row.get("location", "")
        if not url or url.startswith("VRT://"):
            already += 1  # S3 VRT -- rasterio can stream; no local file needed
            continue
        print(f"    Downloading {fname} ...", end=" ", flush=True)
        urllib.request.urlretrieve(url, dest)
        downloaded += 1
        print("done")

    print(f"  [aef] {downloaded} downloaded, {already} already present / streamable")


# -- Step 3: Build cells.parquet ----------------------------------------------

def build_cells(region: str, year: int = 2022, force: bool = False,
                all_cells: bool = False, boundary_path: str | None = None) -> None:
    from puffin.data import build_dataset, build_dataset_allcells, REGION_BBOX

    if all_cells:
        # ALL-CELLS grid: every 100m land cell, pop=0 on empty land -> <region>_all.
        # boundary_path drops cells outside the national border (required for a
        # border-crossing bbox like germany, else foreign towns get labelled 0).
        out_region = f"{region}_all"
        cells_path = DATA_PROCESSED / out_region / "cells.parquet"
        if cells_path.exists() and not force:
            print(f"  [cells] {out_region}: already built ({cells_path})")
            return
        if region not in REGION_BBOX:
            print(f"  [cells] ERROR: ALL_CELLS requires '{region}' in REGION_BBOX")
            return
        print(f"  [cells] Building ALL-CELLS grid for '{region}' -> {out_region} "
              f"(boundary={'yes' if boundary_path else 'NONE'}) ...")
        build_dataset_allcells(region, out_region=out_region, year=year,
                               boundary_path=boundary_path)
        return

    cells_path = DATA_PROCESSED / region / "cells.parquet"

    if region in REGION_BBOX:
        # Known region: build from scratch
        if cells_path.exists() and not force:
            print(f"  [cells] {region}: already built ({cells_path})")
            return
        print(f"  [cells] Building cells.parquet for region={region} ...")
        build_dataset(region=region, year=year)
        return

    # Unknown region bbox: try to filter from a parent region (e.g. germany)
    # To add a new sub-region, add its bbox to REGION_BBOX in data.py instead.
    parent = "germany"
    germany_path = DATA_PROCESSED / parent / "cells.parquet"
    if cells_path.exists() and not force:
        print(f"  [cells] {region}: already built ({cells_path})")
        return
    if not germany_path.exists():
        print(f"  [cells] ERROR: unknown region '{region}' and no {parent}/cells.parquet to filter from.")
        print(f"           Add '{region}' bbox to REGION_BBOX in src/puffin/data.py")
        return

    print(f"  [cells] Filtering {parent} cells to {region} bbox ...")
    import pandas as pd
    bbox = REGION_BBOX.get(region)
    if bbox is None:
        print(f"  [cells] ERROR: '{region}' not in REGION_BBOX -- cannot filter")
        return
    df = pd.read_parquet(germany_path)
    df = df[(df.x_mp >= bbox[0]) & (df.x_mp <= bbox[2]) &
            (df.y_mp >= bbox[1]) & (df.y_mp <= bbox[3])]
    out = DATA_PROCESSED / region
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "cells.parquet", index=False)
    print(f"  [cells] Saved {len(df):,} cells -> {out}/cells.parquet")


# -- Main ---------------------------------------------------------------------

def main() -> None:
    print(f"\n{'='*60}")
    print(f"  Puffin setup [GERMANY]  region={REGION}  year={YEAR}"
          f"{'  [ALL-CELLS]' if ALL_CELLS else ''}")
    print(f"{'='*60}\n")

    if not SKIP_ZENSUS:
        print("Step 1 - Zensus CSVs")
        download_zensus(force=FORCE)

    if not SKIP_AEF:
        print("\nStep 2 - AEF tiles")
        download_aef(REGION, year=YEAR, force=FORCE)

    print("\nStep 3 - Build cells.parquet")
    build_cells(REGION, year=YEAR, force=FORCE, all_cells=ALL_CELLS, boundary_path=BOUNDARY)

    out_name = f"{REGION}_all" if ALL_CELLS else REGION
    print(f"\n{'='*60}")
    print(f"  Setup complete.")
    print(f"  data/processed/{out_name}/cells.parquet is ready.")
    print(f"  Run: python main.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
