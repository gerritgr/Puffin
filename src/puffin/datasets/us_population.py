"""US 2020 census-block population -> data/processed/us_population/cells.parquet.

Source: Census TIGER/Line 2020 block shapefiles (TABBLOCK20), one zip per state. These
bundle geometry + POP20 (2020 population) + INTPTLAT20/INTPTLON20 (a precomputed interior
point), so no separate population API join is needed.

Schema: FIXED region (same as us_voting). Fine unit = census block; coarse unit column is
`county_fips` (GEOID20[:5]) or `tract_fips` (GEOID20[:11]) -- the runner picks it via
cfg.region_col. Target (count): population.

Unlike voting precincts (drawn ~equal-population), census blocks vary from 0 to hundreds
of people, so there is real within-region variance for AlphaEarth to explain -- the direct
US analog of the Germany 100m population grid. Zero-population blocks (parks/water/industry)
are KEPT, mirroring the Germany all-cells build so the model also learns where nobody lives.

AE embedding per block = AE sampled at the block's interior point in EPSG:5070. v1;
area mean-pooling is a later refinement (large rural blocks are under-sampled by a point).

    build_us_population(states=["PA"])          # one state (tiles for PA already local)
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from puffin.config import AE_RAW_DIR, DATA_PROCESSED
from puffin.data import sample_ae_at_points, download_ae_tiles, sample_ae_zonal_mean
from puffin.datasets.us_voting import US_EPSG, US_ZONES

TIGER_URL = ("https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/"
             "tl_2020_{fips}_tabblock20.zip")

# two-letter postal code -> 2-digit state FIPS (50 states + DC)
STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09",
    "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17",
    "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24",
    "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31",
    "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54",
    "WI": "55", "WY": "56",
}


def _download_blocks(fips: str, raw_dir: Path) -> Path:
    """Download + unzip one state's TABBLOCK20 shapefile; return the .shp path."""
    dst_dir = raw_dir / f"tl_2020_{fips}_tabblock20"
    shp = dst_dir / f"tl_2020_{fips}_tabblock20.shp"
    if shp.exists():
        return shp
    url = TIGER_URL.format(fips=fips)
    print(f"  [us_pop] downloading blocks for FIPS {fips}: {url} ...", flush=True)
    with urllib.request.urlopen(url) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(dst_dir)
    if not shp.exists():
        raise FileNotFoundError(f"{shp} not found after unzip")
    return shp


LODES_URL   = "https://lehd.ces.census.gov/data/lodes/LODES8/{st}/wac/{st}_wac_S000_JT00_{yr}.csv.gz"
LODES_YEARS = [2022, 2021, 2020]      # try newest available per state
GVA_CSV     = AE_RAW_DIR.parent / "Gun Violence" / "Gun Violence Archive - DFKI.csv"


def _bin_gun_violence(gdf) -> pd.DataFrame:
    """Bin Gun Violence Archive incident points (lat/lon) to the block that CONTAINS them
    -> DataFrame[unit_id, gun_violence] (incident count per block). Needs block geometry.
    Returns empty (all-0) if the CSV isn't present. Blocks with no incident -> 0."""
    if not GVA_CSV.exists():
        print(f"    gun_violence: CSV not at {GVA_CSV} -> skipping (all 0)", flush=True)
        return pd.DataFrame({"unit_id": pd.Series(dtype=str), "gun_violence": pd.Series(dtype="int64")})
    inc = pd.read_csv(GVA_CSV, usecols=["Latitude", "Longitude"])
    inc = inc.dropna(subset=["Latitude", "Longitude"])
    pts = gpd.GeoDataFrame(inc, geometry=gpd.points_from_xy(inc["Longitude"], inc["Latitude"]),
                           crs=4326).to_crs(gdf.crs)
    j = gpd.sjoin(pts[["geometry"]], gdf[["unit_id", "geometry"]], predicate="within", how="inner")
    cnt = j.groupby("unit_id").size().rename("gun_violence").reset_index()
    print(f"    gun_violence: {len(pts):,} incidents -> {len(cnt):,} blocks with >=1", flush=True)
    return cnt


def _fetch_jobs(states: list[str]) -> pd.DataFrame:
    """LODES8 WAC total jobs (C000) by workplace per 2020 block -> DataFrame[unit_id, jobs].
    Downloads per state (cached; skipped if present). States with no LODES contribute nothing."""
    import urllib.request
    raw = AE_RAW_DIR.parent / "lodes_us"; raw.mkdir(parents=True, exist_ok=True)
    parts, missing = [], []
    for postal in states:
        st = postal.lower(); path = None
        for yr in LODES_YEARS:
            dst = raw / f"{st}_wac_S000_JT00_{yr}.csv.gz"
            if dst.exists():
                path = dst; break
            try:
                with urllib.request.urlopen(LODES_URL.format(st=st, yr=yr), timeout=90) as r:
                    dst.write_bytes(r.read())
                path = dst; break
            except Exception:
                continue
        if path is None:
            missing.append(postal); continue
        w = pd.read_csv(path, compression="gzip", usecols=["w_geocode", "C000"], dtype={"w_geocode": str})
        parts.append(w.rename(columns={"w_geocode": "unit_id", "C000": "jobs"})[["unit_id", "jobs"]])
        print(f"    LODES {postal}: {len(w):,} blocks with jobs", flush=True)
    if missing:
        print(f"    LODES unavailable (jobs=0): {missing}", flush=True)
    if not parts:
        return pd.DataFrame({"unit_id": pd.Series(dtype=str), "jobs": pd.Series(dtype="int64")})
    jz = pd.concat(parts, ignore_index=True)
    jz["jobs"] = pd.to_numeric(jz["jobs"], errors="coerce").fillna(0).astype("int64")
    return jz.groupby("unit_id", as_index=False)["jobs"].sum()


def load_us_blocks(states: list[str], raw_dir: Path | None = None,
                   with_geometry: bool = False) -> pd.DataFrame:
    """Load census blocks for the given states -> DataFrame with unit_id, county_fips,
    tract_fips, x_mp/y_mp (EPSG:5070 interior point), population. If with_geometry, returns
    a GeoDataFrame keeping the block polygon (needed for area-pooled AE). No AE yet."""
    raw_dir = raw_dir or (AE_RAW_DIR.parent / "census_us")
    raw_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for st in states:
        fips = STATE_FIPS.get(st.upper())
        if fips is None:
            raise KeyError(f"unknown state postal code {st!r}")
        shp = _download_blocks(fips, raw_dir)
        cols = ["GEOID20", "POP20", "HOUSING20", "INTPTLAT20", "INTPTLON20"]
        gdf = gpd.read_file(shp, columns=cols, engine="pyogrio", read_geometry=with_geometry)
        print(f"  [us_pop] {st} (FIPS {fips}): {len(gdf):,} blocks", flush=True)
        frames.append(gdf)
    blk = pd.concat(frames, ignore_index=True)

    geoid = blk["GEOID20"].astype(str)
    lat = pd.to_numeric(blk["INTPTLAT20"], errors="coerce").values
    lon = pd.to_numeric(blk["INTPTLON20"], errors="coerce").values
    from pyproj import Transformer
    x, y = Transformer.from_crs(4326, US_EPSG, always_xy=True).transform(lon, lat)

    df = pd.DataFrame({
        "unit_id":     geoid.values,                 # block GEOID (fine unit)
        "county_fips": geoid.str[:5].values,         # region_id for County -> block
        "tract_fips":  geoid.str[:11].values,        # region_id for Tract  -> block
        "x_mp":        np.rint(x).astype("int64"),
        "y_mp":        np.rint(y).astype("int64"),
        "population":  pd.to_numeric(blk["POP20"],     errors="coerce").fillna(0).astype("int64").values,
        "housing":     pd.to_numeric(blk["HOUSING20"], errors="coerce").fillna(0).astype("int64").values,
    })
    if with_geometry:
        df = gpd.GeoDataFrame(df, geometry=blk.geometry.values, crs=frames[0].crs)
    df = df[np.isfinite(df["x_mp"]) & np.isfinite(df["y_mp"])].reset_index(drop=True)
    return df


def build_us_population(states: list[str] | None = None, out_region: str = "us_population",
                        raw_dir: Path | None = None, prefetch: bool = True,
                        pool: str = "zonal", ae_year: int = 2020) -> pd.DataFrame:
    """Assemble cells.parquet for the requested states (census blocks + AE features).

    ae_year=2020 (default): AlphaEarth vintage. Blocks + population + housing are the 2020
    Census, so 2020 matches; jobs (LODES ~2022) and gun violence are close eras. (Voting uses
    AE 2024 to match the 2024 election -- that is a separate build.)
    pool="zonal" (default): AE per block = MEAN over all 100m cells inside the block polygon
    (the correct grid analog). pool="point": single AE sample at the block interior point.
    prefetch=True: download+resample covering AE tiles to local 100m COGs first (fast)."""
    if states is None:
        states = list(STATE_FIPS.keys())
    df = load_us_blocks(states, raw_dir=raw_dir, with_geometry=(pool == "zonal"))
    print(f"  [us_pop] {len(df):,} blocks over {df['county_fips'].nunique():,} counties, "
          f"total pop {df['population'].sum()/1e6:.1f}M  housing {df['housing'].sum()/1e6:.1f}M", flush=True)

    # jobs (LODES workplace) -> join by block GEOID; blocks with no jobs -> 0
    print(f"  [us_pop] fetching LODES jobs ...", flush=True)
    df["unit_id"] = df["unit_id"].astype(str)
    df = df.merge(_fetch_jobs(states), on="unit_id", how="left")
    df["jobs"] = df["jobs"].fillna(0).astype("int64")
    print(f"  [us_pop] jobs total {df['jobs'].sum()/1e6:.1f}M", flush=True)

    # gun violence (GVA incident points binned to the containing block; needs geometry)
    if pool == "zonal":
        print(f"  [us_pop] binning gun-violence incidents ...", flush=True)
        df = df.merge(_bin_gun_violence(df), on="unit_id", how="left")
        df["gun_violence"] = df["gun_violence"].fillna(0).astype("int64")
        print(f"  [us_pop] gun_violence total {int(df['gun_violence'].sum()):,} incidents", flush=True)
    else:
        df["gun_violence"] = 0

    if pool == "zonal":
        # Poisson exposure = block AREA in units of 100m cells (area_m2 / 100^2). Lets the
        # model predict density from AE and scale it to a count: rate = area * density.
        areas = df.geometry.to_crs(US_EPSG).area.values
        df["exposure"] = (areas / 1e4).astype("float32")

    if prefetch:
        print(f"  [us_pop] prefetching AE tiles for {len(df):,} blocks ...", flush=True)
        download_ae_tiles(df, year=ae_year, target_epsg=US_EPSG, zones=US_ZONES)

    if pool == "zonal":
        print(f"  [us_pop] area-pooling AE over {len(df):,} block polygons ...", flush=True)
        df = sample_ae_zonal_mean(df, year=ae_year, target_epsg=US_EPSG, zones=US_ZONES)
    else:
        print(f"  [us_pop] point-sampling AE at {len(df):,} block interior points ...", flush=True)
        df = sample_ae_at_points(df, year=ae_year, target_epsg=US_EPSG, zones=US_ZONES)

    out = DATA_PROCESSED / out_region
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "cells.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  [us_pop] saved -> {out_path}  ({len(df):,} blocks, "
          f"{df['county_fips'].nunique():,} counties, {df['tract_fips'].nunique():,} tracts; "
          f"pop {df['population'].sum()/1e6:.1f}M  housing {df['housing'].sum()/1e6:.1f}M  "
          f"jobs {df['jobs'].sum()/1e6:.1f}M  gun_violence {int(df['gun_violence'].sum()):,})")
    return df
