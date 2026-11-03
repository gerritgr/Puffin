"""US 2024 presidential precinct voting -> data/processed/us_voting/cells.parquet.

Source: NYT `presidential-precinct-map-2024` (national precincts + results).
Schema: FIXED region. Fine unit = precinct; coarse unit column is `county_fips`
(GEOID[:5]) or `state_fips` (GEOID[:2]) -- the runner picks it via cfg.region_col.
Targets (count): votes_rep (Trump), votes_dem (Harris), votes_total, votes_other.

AE embedding per precinct (pool="zonal", default) = MEAN AE over all 100m cells inside the
precinct polygon, and `exposure` = precinct area in units of 100m cells -- exactly as for the
US census blocks (datasets/us_population.py). pool="point" keeps the old v1 behaviour (one AE
sample at the precinct's representative point, no area).

Running the full nation streams thousands of AE tiles from S3 (heavy) -> do it on
the cluster. Pass `states=[...]` (postal codes) to build one/few states first.
"""
from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from puffin.config import AE_RAW_DIR, DATA_PROCESSED
from puffin.data import sample_ae_at_points, sample_ae_zonal_mean, download_ae_tiles

# NYT national files (see repo README). TopoJSON is listed first because it is the one
# NYT actually serves; the GeoJSON path 404s and is kept only as a fallback in case they
# restore it.
NYT_URLS = [
    "https://int.nyt.com/newsgraphics/elections/map-data/2024/national/precincts-with-results.topojson.gz",
    "https://int.nyt.com/newsgraphics/elections/map-data/2024/national/precincts-with-results.geojson.gz",
]
US_EPSG  = 5070                                  # CONUS Albers Equal Area (metres)
US_ZONES = tuple(f"{z}N" for z in range(10, 20)) # CONUS UTM zones 10N..19N (bbox still finds others)


def _download(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Use an already-downloaded file before touching the network, so a rerun never
    # re-requests (and never re-reports the dead GeoJSON URL as a 404).
    for ext in (".topojson", ".geojson"):
        dst = raw_dir / f"precincts-with-results{ext}"
        if dst.exists():
            print(f"  [us_voting] using cached {dst.name} ({dst.stat().st_size/1e6:.0f} MB)")
            return dst

    for url in NYT_URLS:
        ext = ".geojson" if "geojson" in url else ".topojson"
        dst = raw_dir / f"precincts-with-results{ext}"
        try:
            print(f"  [us_voting] downloading {url} ...", flush=True)
            with urllib.request.urlopen(url) as r:
                data = gzip.decompress(r.read())
            dst.write_bytes(data)
            return dst
        except Exception as e:
            print(f"    failed ({e}); trying next source")
    raise FileNotFoundError("Could not download NYT precinct data from any known URL.")


def load_us_precincts(year: int = 2024, states: list[str] | None = None,
                      raw_dir: Path | None = None, with_geometry: bool = False) -> pd.DataFrame:
    """Load NYT precincts (optionally one/few states) -> DataFrame with unit_id,
    county_fips, state_fips, x_mp/y_mp (EPSG:5070 interior point) and vote counts.
    This is everything EXCEPT the AE sampling, so it can drive tile prefetch too.
    If with_geometry, returns a GeoDataFrame (EPSG:5070) keeping the precinct polygon, as
    needed for area-pooled AE and the area exposure."""
    raw_dir = raw_dir or (AE_RAW_DIR.parent / "voting_us")
    src = _download(raw_dir)

    # Reading the 611MB national TopoJSON is the memory peak: OGR expands topology arcs
    # into ~150k full precinct geometries. When only some states are wanted, push the
    # filter down to OGR so the rest are never materialised in Python (tens of GB saved).
    gdf = None
    if states:
        where = "state IN (" + ",".join(f"'{s}'" for s in states) + ")"
        try:
            print(f"  [us_voting] reading {src.name}  [pushdown: {where}] ...", flush=True)
            gdf = gpd.read_file(src, engine="pyogrio", where=where)
        except Exception as e:
            print(f"    pushdown filter unavailable ({type(e).__name__}: {e}); "
                  f"reading all precincts instead (needs much more RAM)", flush=True)
            gdf = None
    if gdf is None:
        print(f"  [us_voting] reading {src.name} ...", flush=True)
        try:
            gdf = gpd.read_file(src, engine="pyogrio")   # lower peak memory + faster than fiona
        except Exception as e:
            print(f"    pyogrio unavailable ({e}); falling back to default engine (needs more RAM)")
            gdf = gpd.read_file(src)
    need = {"GEOID", "votes_dem", "votes_rep", "votes_total"}
    missing = need - set(gdf.columns)
    if missing:
        raise KeyError(f"NYT data is missing expected fields {missing}; got {list(gdf.columns)}")

    # Filter to requested states BEFORE the expensive .copy() so we never duplicate the full
    # national frame. Reading the 611MB national TopoJSON is the memory peak; the PA subset is tiny.
    if states is not None and "state" in gdf.columns:
        gdf = gdf[gdf["state"].isin(states)]
        print(f"  [us_voting] filtered to states {states}: {len(gdf):,} precincts")
    gdf = gdf[gdf.geometry.notna() & gdf["GEOID"].notna()].copy()

    if gdf.crs is None:
        gdf = gdf.set_crs(4326, allow_override=True)   # NYT precincts are WGS84 lon/lat
    gdf = gdf.to_crs(US_EPSG)
    if with_geometry:
        # TopoJSON-derived polygons can be self-intersecting; repair them so the area and the
        # rasterized zonal mean are correct (valid polygons are left untouched).
        bad = ~gdf.geometry.is_valid
        if bad.any():
            try:
                gdf.loc[bad, "geometry"] = gdf.loc[bad, "geometry"].make_valid()
            except AttributeError:
                gdf.loc[bad, "geometry"] = gdf.loc[bad, "geometry"].buffer(0)
            print(f"  [us_voting] repaired {int(bad.sum()):,} invalid precinct geometries", flush=True)
    pt = gdf.geometry.representative_point()      # a guaranteed-interior point per precinct

    geoid = gdf["GEOID"].astype(str)
    df = pd.DataFrame({
        "unit_id":     geoid.values,                 # precinct GEOID (fine unit)
        "county_fips": geoid.str[:5].values,         # region_id for County -> Precinct
        "state_fips":  geoid.str[:2].values,         # region_id for State  -> Precinct
        "x_mp":        np.rint(pt.x.values).astype("int64"),
        "y_mp":        np.rint(pt.y.values).astype("int64"),
        "votes_dem":   pd.to_numeric(gdf["votes_dem"],   errors="coerce").fillna(0).astype("int64").values,
        "votes_rep":   pd.to_numeric(gdf["votes_rep"],   errors="coerce").fillna(0).astype("int64").values,
        "votes_total": pd.to_numeric(gdf["votes_total"], errors="coerce").fillna(0).astype("int64").values,
    })
    if "state" in gdf.columns:
        df["state"] = gdf["state"].astype(str).values
    df["votes_other"] = (df["votes_total"] - df["votes_dem"] - df["votes_rep"]).clip(lower=0)
    if with_geometry:
        df = gpd.GeoDataFrame(df, geometry=gdf.geometry.values, crs=US_EPSG)
    return df


def build_us_voting(year: int = 2024, out_region: str = "us_voting",
                    states: list[str] | None = None,
                    raw_dir: Path | None = None,
                    prefetch: bool = True,
                    pool: str = "zonal") -> pd.DataFrame:
    """Assemble cells.parquet for the requested states.

    prefetch=True (default, the "Germany way"): download the covering AE tiles locally
    first, then sample OFFLINE from those files -- robust over a flaky connection.
    prefetch=False: sample by streaming each tile directly (fine on a fast/stable link
    like the cluster in us-west-2).
    pool="zonal" (default): AE per precinct = MEAN over all 100m cells in the precinct polygon,
    plus `exposure` = precinct area in 100m cells (same as the census blocks).
    pool="point": single AE sample at the precinct interior point, no exposure (old v1)."""
    if pool not in ("zonal", "point"):
        raise ValueError(f"pool must be 'zonal' or 'point', got {pool!r}")
    df = load_us_precincts(year=year, states=states, raw_dir=raw_dir, with_geometry=(pool == "zonal"))

    if pool == "zonal":
        # exposure = precinct AREA in units of 100m cells (area_m2 / 100^2), exactly as for the
        # census blocks: the model predicts a density from AE and area scales it to a count.
        df["exposure"] = (df.geometry.area.values / 1e4).astype("float32")
        print(f"  [us_voting] precinct area (100m cells): median {np.median(df['exposure']):.1f}, "
              f"max {df['exposure'].max():,.0f}", flush=True)

    if prefetch:
        print(f"  [us_voting] prefetching AE tiles for {len(df):,} precincts ...", flush=True)
        download_ae_tiles(df, year=year, target_epsg=US_EPSG, zones=US_ZONES)

    if pool == "zonal":
        print(f"  [us_voting] area-pooling AE over {len(df):,} precinct polygons ...", flush=True)
        df = sample_ae_zonal_mean(df, year=year, target_epsg=US_EPSG, zones=US_ZONES)
    else:
        print(f"  [us_voting] sampling AE at {len(df):,} precinct points (EPSG:{US_EPSG}) ...", flush=True)
        df = sample_ae_at_points(df, year=year, target_epsg=US_EPSG, zones=US_ZONES)

    out = DATA_PROCESSED / out_region
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "cells.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  [us_voting] saved -> {out_path}  ({len(df):,} precincts, "
          f"{df['county_fips'].nunique():,} counties, {df['state_fips'].nunique():,} states, "
          f"total votes {df['votes_total'].sum()/1e6:.1f}M)")
    return df
