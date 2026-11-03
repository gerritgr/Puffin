"""Data pipeline: load -> align -> aggregate -> split.

Public API
----------
load_zensus(variable, bbox_epsg3035)        -> DataFrame (grid cells + value)
sample_ae_at_zensus_cells(zensus_df, year)  -> DataFrame (+ 64 AE columns)
build_dataset(region, year, normalize)      -> DataFrame saved as parquet
build_coarse_blocks(fine_df, region_m, target)-> block-level DataFrame
make_spatial_splits(block_df, scheme, seed) -> dict {fold: [block_ids]}

CoarseBlockDataset, collate_variable_blocks -> PyTorch training utilities

CRS rule: everything is EPSG:3035 after loading. Zensus CSVs are natively
EPSG:3035. AE tiles are UTM (32631/32632/32633) and are warped on load.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds, rowcol
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
from shapely.geometry import box
from torch import Tensor
from torch.utils.data import Dataset

from puffin.config import (
    AE_DIM, AE_RAW_DIR, COUNT_TARGETS, DATA_PROCESSED, EPSG_INSPIRE, FINE_RES_M,
    ZENSUS_RAW_DIR,
)

# ---------------------------------------------------------------------------
# Region bounding boxes in EPSG:3035 (west, south, east, north) in metres
# Derived from Zensus population data extents plus small margin.
# ---------------------------------------------------------------------------
REGION_BBOX: dict[str, tuple[int, int, int, int]] = {
    "berlin":  (4468000, 3196000, 4570000, 3292000),
    "leipzig": (4430000, 3090000, 4490000, 3140000),
    # further German cities, kept so a bounding box can be built for each
    "munich":    (4256000, 2738000, 4340000, 2800000),
    "dresden":   (4573000, 3099000, 4598000, 3123000),
    "nuremberg": (4390000, 2910000, 4408000, 2939000),
    "hannover":  (4293000, 3243000, 4315000, 3266000),
    "germany": (4031000, 2689000, 4672000, 3548000),
}

# ---------------------------------------------------------------------------
# Zensus variable metadata
# ---------------------------------------------------------------------------
ZENSUS_META: dict[str, dict] = {
    "population": {
        "file":    "population/Zensus2022_Bevoelkerungszahl_100m-Gitter.csv",
        "col":     "Einwohner",
        "sep":     ";",
        "enc":     "utf-8",
        "decimal": ".",
        "dtype":   "int32",
    },
    "rent_per_m2": {
        "file":    "rent/Zensus2022_Durchschn_Nettokaltmiete_100m-Gitter.csv",
        "col":     "durchschnMieteQM",
        "sep":     ";",
        "enc":     "utf-8",
        "decimal": ",",
        "dtype":   "float32",
    },
    "avg_age": {
        "file":    "age/Zensus2022_Durchschnittsalter_100m-Gitter.csv",
        "col":     "Durchschnittsalter",
        "sep":     ";",
        "enc":     "utf-8",
        "decimal": ",",
        "dtype":   "float32",
    },
    "vacancy_rate": {
        "file":    "vacancy/Zensus2022_Leerstandsquote_100m-Gitter.csv",
        "col":     "Leerstandsquote",
        "sep":     ";",
        "enc":     "utf-8",
        "decimal": ",",
        "dtype":   "float32",
        "scale":   0.01,   # Zensus stores as percentage (0-100); convert to proportion (0-1)
    },
    "foreigner_share": {
        "file":    "foreigner_share/Zensus2022_Anteil_Auslaender_100m-Gitter.csv",
        "col":     "AnteilAuslaender",
        "sep":     ";",
        "enc":     "utf-8",
        "decimal": ",",
        "dtype":   "float32",
    },
    "citizens_18plus": {
        "file":    "citizens_18plus/Deutsche_Staatsangehoerige_ab_18_Jahren/Zensus2022_Deutsche_Staatsangehoerige_ab18_100m-Gitter.csv",
        "col":     None,   # auto-detected (last numeric col before flag col)
        "sep":     ";",
        "enc":     "utf-8",
        "decimal": ".",
        "dtype":   "float32",
    },
}

EPS = 1e-6


# ============================================================
# 1. Load Zensus CSV
# ============================================================

def load_zensus(
    variable: str,
    bbox_epsg3035: tuple[int, int, int, int] | None = None,
) -> pd.DataFrame:
    """Parse a Zensus 2022 100m CSV for one variable.

    Parameters
    ----------
    variable      : one of the keys in ZENSUS_META
    bbox_epsg3035 : (west, south, east, north) in EPSG:3035 metres to filter.
                    None -> return all of Germany.

    Returns
    -------
    DataFrame with columns: grid_id, x_mp, y_mp, <variable>
    x_mp/y_mp are the cell centroid coordinates in EPSG:3035 (metres).

    Missing-data rules (Logic Module §3.2):
    - Population: cells absent from CSV -> 0 (assumed uninhabited).
    - Rate variables (rent, age, vacancy): absent cells -> NaN (undefined).
    - KLAMMERN flag (disclosure-control noise): kept as-is; τ_B² accounts for it.
    """
    meta = ZENSUS_META[variable]
    path = ZENSUS_RAW_DIR / meta["file"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/02_download_zensus.py"
        )

    df = pd.read_csv(
        path,
        sep=meta["sep"],
        encoding=meta["enc"],
        usecols=lambda c: c not in ("werterlaeuternde_Zeichen",),
        dtype=str,          # read everything as string; we handle types below
        low_memory=False,
    )

    # Auto-detect value column if not specified
    val_col = meta["col"]
    if val_col is None:
        numeric_cols = [c for c in df.columns
                        if c not in ("GITTER_ID_100m", "x_mp_100m", "y_mp_100m")]
        val_col = numeric_cols[-1]

    df = df[["GITTER_ID_100m", "x_mp_100m", "y_mp_100m", val_col]].rename(columns={
        "GITTER_ID_100m": "grid_id",
        "x_mp_100m":      "x_mp",
        "y_mp_100m":      "y_mp",
        val_col:          variable,
    })

    # Cast coordinate columns
    df["x_mp"] = pd.to_numeric(df["x_mp"], errors="coerce").astype("int32")
    df["y_mp"] = pd.to_numeric(df["y_mp"], errors="coerce").astype("int32")

    # Cast value column: replace comma decimal separator, then coerce to numeric
    # (em-dash '–' and other non-numeric flags become NaN via errors='coerce')
    val_str = df[variable].astype(str)
    if meta["decimal"] == ",":
        val_str = val_str.str.replace(",", ".", regex=False)
    df[variable] = pd.to_numeric(val_str, errors="coerce").astype("float32")

    # Apply optional unit-conversion scale (e.g. percentage → proportion)
    if "scale" in meta:
        df[variable] = df[variable] * meta["scale"]

    # Filter to bbox
    if bbox_epsg3035 is not None:
        west, south, east, north = bbox_epsg3035
        df = df[
            (df.x_mp >= west)  & (df.x_mp <= east) &
            (df.y_mp >= south) & (df.y_mp <= north)
        ].copy()

    return df.reset_index(drop=True)


# ============================================================
# 2. Sample AE embeddings at Zensus cell centroids
# ============================================================

def _find_ae_tiles(bbox: tuple, year: int, target_epsg: int = EPSG_INSPIRE,
                   zones: tuple | None = None) -> list:
    """Return tile paths (local Path or remote string) intersecting the bbox.

    Country-agnostic: `bbox` is (west, south, east, north) in `target_epsg`
    (Germany EPSG:3035 by default, US EPSG:5070, ...). `zones` optionally restricts
    the UTM zones searched (None -> any zone the bbox hits, from the global index).

    Priority:
      1. Local .tiff files in data/raw/aef/<year>/<zone>/ (fast, offline)
      2. Remote VRT/S3 paths from aef_index.parquet (streams over network -
         requires GDAL with /vsis3 support and AWS credentials or public bucket)

    Set env var AEF_FORCE_REMOTE=1 to skip local files and always use S3.
    """
    import os
    from pyproj import Transformer

    west, south, east, north = bbox
    t = Transformer.from_crs(target_epsg, 4326, always_xy=True)
    # transform all 4 corners (robust to projection skew), take the enclosing lon/lat box
    lons, lats = t.transform([west, east, west, east], [south, south, north, north])
    lon_w, lon_e = min(lons), max(lons)
    lat_s, lat_n = min(lats), max(lats)

    index_path = AE_RAW_DIR / "aef_index.parquet"
    hits = None
    zone_set = None
    if index_path.exists():
        idx = pd.read_parquet(index_path)
        m = ((idx.year == year) &
             (idx.wgs84_east  >= lon_w) & (idx.wgs84_west  <= lon_e) &
             (idx.wgs84_north >= lat_s) & (idx.wgs84_south <= lat_n))
        if zones is not None:
            m &= idx.utm_zone.isin(list(zones))
        hits = idx[m]
        stem_set = {Path(p).stem for p in hits["path"]}
        if "utm_zone" in hits.columns:
            zone_set = sorted(hits["utm_zone"].unique())
    else:
        stem_set = None

    # ── 1. Try local files first ──────────────────────────────────────────
    force_remote = os.environ.get("AEF_FORCE_REMOTE", "0") == "1"
    local_tiles: list[Path] = []
    if not force_remote:
        year_dir = AE_RAW_DIR / str(year)
        # zones to search: explicit -> given; else the zones the bbox hits; else all local dirs
        search_zones = (list(zones) if zones is not None
                        else zone_set if zone_set is not None
                        else [d.name for d in year_dir.glob("*") if d.is_dir()])
        for zone in search_zones:
            zone_dir = year_dir / str(zone)
            if not zone_dir.exists():
                continue
            for f in zone_dir.glob("*_100m.tiff"):
                stem = f.stem.replace("_100m", "")
                if stem_set is None or stem in stem_set:
                    local_tiles.append(f)

    if local_tiles:
        return local_tiles

    # ── 2. Stream from the PUBLIC source.coop bucket via /vsicurl ─────────
    # The tiles live on a public open-data bucket, so we read them over HTTPS with
    # windowed reads (only the pixels each point needs) -- no AWS creds, no full
    # download. This is how AlphaEarth is meant to be consumed at continent scale
    # (a single 10m tile is ~3.7 GB; a country like the US is terabytes).
    #   s3://<bucket>/<rest>  ->  /vsicurl/https://data.source.coop/<rest>
    if hits is not None and len(hits) > 0 and "path" in hits.columns:
        os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
        os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tiff")
        os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
        os.environ.setdefault("VSI_CACHE", "TRUE")
        remote = []
        for p in hits["path"].dropna():
            if isinstance(p, str) and p.startswith("s3://"):
                rest = p.split("/", 3)[-1]                     # after s3://<bucket>/
                remote.append(f"/vsicurl/https://data.source.coop/{rest}")
        if remote:
            print(f"  [AEF] no local tiles -> streaming {len(remote)} tiles via /vsicurl "
                  f"(public bucket, windowed reads)")
            return remote

    raise FileNotFoundError(
        f"No AE tiles found for year={year} covering bbox {bbox} (EPSG:{target_epsg}). "
        "Either download tiles locally under data/raw/aef/<year>/<zone>/, or ensure "
        "aef_index.parquet has 'location' (S3 VRT) paths and GDAL /vsis3 is configured."
    )


def _warp_tile_to_epsg3035(
    tile_path: Path,
    target_transform,
    target_width: int,
    target_height: int,
    target_crs: CRS,
) -> np.ndarray:
    """Warp one 64-band UTM tile onto the target EPSG:3035 raster grid.

    Returns array of shape (64, target_height, target_width), float32.
    Pixels not covered by this tile are NaN.
    Uses Resampling.nearest to preserve integer embedding values.
    """
    with rasterio.open(tile_path) as src:
        dest = np.full(
            (src.count, target_height, target_width), np.nan, dtype=np.float32
        )
        reproject(
            source=rasterio.band(src, list(range(1, src.count + 1))),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
    return dest


def sample_ae_at_zensus_cells(
    zensus_df: pd.DataFrame,
    year: int = 2022,
    normalize: str = "R0",
    target_epsg: int = EPSG_INSPIRE,
    zones: tuple | None = None,
) -> pd.DataFrame:
    """Sample 64-dim AE embeddings at each cell/point centroid.

    Warps AE tiles from UTM -> `target_epsg`, mosaics them tile-by-tile, then
    reads the pixel value at each point's (x_mp, y_mp) centroid. Country-agnostic:
    Germany uses EPSG:3035 (default); US voting uses EPSG:5070; `zones` optionally
    restricts the UTM zones searched (None -> any zone the bbox hits).

    Parameters
    ----------
    zensus_df   : DataFrame with x_mp, y_mp in `target_epsg` metres
    year        : AEF vintage year
    normalize   : "R0" no-op | "R1" per-cell L2 | "R2" per-dim z-score
    target_epsg : projected CRS of x_mp/y_mp and of the warped mosaic
    zones       : UTM zones to search (None -> derived from the bbox)

    Returns
    -------
    Input DataFrame with 64 columns ae_0 … ae_63 added.
    Rows where AE is missing (outside all tiles) are dropped.
    """
    if zensus_df.empty:
        raise ValueError("zensus_df is empty")

    target_crs = CRS.from_epsg(target_epsg)

    # Find bounding box of all cells to locate relevant AEF tiles
    x_min_all = (int(zensus_df.x_mp.min()) // FINE_RES_M - 1) * FINE_RES_M
    x_max_all = (int(zensus_df.x_mp.max()) // FINE_RES_M + 2) * FINE_RES_M
    y_min_all = (int(zensus_df.y_mp.min()) // FINE_RES_M - 1) * FINE_RES_M
    y_max_all = (int(zensus_df.y_mp.max()) // FINE_RES_M + 2) * FINE_RES_M
    bbox = (x_min_all, y_min_all, x_max_all, y_max_all)

    tiles = _find_ae_tiles(bbox, year, target_epsg=target_epsg, zones=zones)
    if not tiles:
        raise FileNotFoundError(
            f"No AE tiles found for year={year} covering bbox {bbox}. "
            "Run: python scripts/01_download_alphaearth.py"
        )

    # x_mp, y_mp are EPSG:3035 centroids (centre of each 100m cell)
    xs = zensus_df.x_mp.values.astype(float)
    ys = zensus_df.y_mp.values.astype(float)

    print(f"  Sampling {len(tiles)} AE tile(s) tile-by-tile "
          f"({len(zensus_df):,} cells) ...")

    # Process one tile at a time to avoid allocating a full-region mosaic.
    # Each tile gets its own sub-extent: only pixels needed for that tile's cells.
    embeddings = np.full((len(zensus_df), AE_DIM), np.nan, dtype=np.float32)
    margin = 2 * FINE_RES_M

    for i, tile in enumerate(tiles):
        with rasterio.open(tile) as _src:
            tile_bounds_3035 = transform_bounds(_src.crs, target_crs, *_src.bounds)

        tx0, ty0, tx1, ty1 = tile_bounds_3035
        in_tile = ((xs >= tx0 - margin) & (xs <= tx1 + margin) &
                   (ys >= ty0 - margin) & (ys <= ty1 + margin))
        n_in = in_tile.sum()
        if n_in == 0:
            continue

        sub_xs = xs[in_tile]
        sub_ys = ys[in_tile]

        sub_x_min = (int(sub_xs.min()) // FINE_RES_M - 1) * FINE_RES_M
        sub_x_max = (int(sub_xs.max()) // FINE_RES_M + 2) * FINE_RES_M
        sub_y_min = (int(sub_ys.min()) // FINE_RES_M - 1) * FINE_RES_M
        sub_y_max = (int(sub_ys.max()) // FINE_RES_M + 2) * FINE_RES_M

        sub_w = (sub_x_max - sub_x_min) // FINE_RES_M
        sub_h = (sub_y_max - sub_y_min) // FINE_RES_M

        sub_transform = rasterio.transform.from_origin(
            west=sub_x_min, north=sub_y_max, xsize=FINE_RES_M, ysize=FINE_RES_M
        )

        mb = AE_DIM * sub_w * sub_h * 4 / 1e6
        print(f"    [{i+1}/{len(tiles)}] {n_in:,} cells, "
              f"sub-mosaic {sub_w}×{sub_h} px ({mb:.0f} MB)")

        warped = _warp_tile_to_epsg3035(tile, sub_transform, sub_w, sub_h, target_crs)

        r_idx, c_idx = rowcol(sub_transform, sub_xs, sub_ys)
        r_idx = np.clip(r_idx, 0, sub_h - 1)
        c_idx = np.clip(c_idx, 0, sub_w - 1)

        emb = warped[:, r_idx, c_idx].T  # (n_in, 64)
        del warped

        valid_emb = ~np.isnan(emb).all(axis=1)
        global_idx = np.where(in_tile)[0][valid_emb]
        embeddings[global_idx] = emb[valid_emb]

    ae_cols = [f"ae_{d}" for d in range(AE_DIM)]
    ae_df = pd.DataFrame(embeddings, columns=ae_cols, index=zensus_df.index)

    # Drop cells where AE is all-NaN (outside tile coverage)
    valid_mask = ~np.isnan(embeddings).all(axis=1)
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} cells with missing AE coverage.")

    result = pd.concat([zensus_df[valid_mask].reset_index(drop=True),
                        ae_df[valid_mask].reset_index(drop=True)], axis=1)

    # Apply normalization
    if normalize == "R1":
        norms = np.linalg.norm(result[ae_cols].values, axis=1, keepdims=True)
        result[ae_cols] = result[ae_cols].values / np.maximum(norms, EPS)
    elif normalize == "R2":
        raise NotImplementedError(
            "R2 normalization requires fitting on training split - "
            "implement after splits are created."
        )
    # R0: no-op (raw values; LayerNorm inside MLPHead handles scale)

    return result


def sample_ae_at_points(
    df: pd.DataFrame,
    year: int,
    target_epsg: int,
    zones: tuple | None = None,
) -> pd.DataFrame:
    """Fast AE sampling at SPARSE points (x_mp, y_mp in target_epsg metres).

    Unlike sample_ae_at_zensus_cells (which warps a sub-mosaic per tile -- built for a
    dense grid and slow to STREAM), this assigns each point to the tile that contains
    it via the index bbox, opens ONLY tiles that actually hold points, and reads just
    the 64-band pixel value at each point. Ideal for us_voting over the public bucket.
    Rows with no AE coverage are dropped. Returns df (filtered) + ae_0..ae_63.
    """
    import os, random, socket, time
    from pyproj import Transformer

    def _online(host="s3.us-west-2.amazonaws.com", port=443, timeout=5) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def _wait_online(max_wait=1800, step=10):
        """Block until S3 is reachable again (fluctuating connection)."""
        waited = 0
        while not _online() and waited < max_wait:
            print(f"    offline; waiting for S3 to come back ... ({waited}s)", flush=True)
            time.sleep(step); waited += step

    if df.empty:
        raise ValueError("df is empty")
    idx_path = AE_RAW_DIR / "aef_index.parquet"
    if not idx_path.exists():
        raise FileNotFoundError(f"{idx_path} not found")
    idx = pd.read_parquet(idx_path)

    xs = df["x_mp"].values.astype(float)
    ys = df["y_mp"].values.astype(float)
    lon, lat = Transformer.from_crs(target_epsg, 4326, always_xy=True).transform(xs, ys)
    lon = np.asarray(lon); lat = np.asarray(lat)

    m = (idx.year == year)
    if zones is not None:
        m &= idx.utm_zone.isin(list(zones))
    m &= ((idx.wgs84_east >= lon.min()) & (idx.wgs84_west <= lon.max()) &
          (idx.wgs84_north >= lat.min()) & (idx.wgs84_south <= lat.max()))
    hits = idx[m].reset_index(drop=True)
    if hits.empty:
        raise FileNotFoundError(f"No AE tiles for year={year} covering the points.")

    # Stream from the direct S3 backend (/vsis3), NOT the data.source.coop Cloudflare
    # front: Cloudflare throttles bursts of range reads (429/520) and blocks non-browser
    # user-agents, which made streaming and whole-file downloads unreliable. The bucket
    # is public, so no-sign-request reads it anonymously straight from AWS us-west-2.
    for k, v in {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
                 "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tiff",
                 "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
                 "VSI_CACHE": "TRUE", "GDAL_CACHEMAX": "512",
                 "AWS_NO_SIGN_REQUEST": "YES", "AWS_REGION": "us-west-2",
                 "GDAL_HTTP_MAX_RETRY": "5",    # retry flaky/partial range reads
                 "GDAL_HTTP_RETRY_DELAY": "5",  # seconds between GDAL-level retries
                 "GDAL_HTTP_RETRY_CODES": "429,500,502,503,504,520,522,524",
                 "GDAL_HTTP_TIMEOUT": "120"}.items():
        os.environ.setdefault(k, v)

    def _tile_src(row) -> str:
        stem = Path(row["path"]).stem
        for name in (f"{stem}_100m.tiff", f"{stem}.tiff"):
            local = AE_RAW_DIR / str(year) / str(row["utm_zone"]) / name
            if local.exists():
                return str(local)
        # index path is s3://<bucket>/<key> -> /vsis3/<bucket>/<key> (direct S3)
        return str(row["path"]).replace("s3://", "/vsis3/", 1)

    MAX_TILE_ATTEMPTS = 8     # with real backoff, 8 tries spans ~10 min of throttling
    BACKOFF_BASE = 5.0        # seconds; doubles each attempt (5,10,20,40,80,160,...)
    BACKOFF_CAP  = 180.0
    TILE_PAUSE   = 1.0        # polite gap between tiles so we don't trip the rate limit
    n = len(df)
    AE = np.full((n, AE_DIM), np.nan, dtype=np.float32)
    assigned = np.zeros(n, dtype=bool)
    print(f"  sampling AE at {n:,} points from up to {len(hits)} candidate tiles ...", flush=True)

    for _, row in hits.iterrows():
        inb = ((~assigned) &
               (lon >= row["wgs84_west"]) & (lon <= row["wgs84_east"]) &
               (lat >= row["wgs84_south"]) & (lat <= row["wgs84_north"]))
        gi = np.where(inb)[0]
        if gi.size == 0:
            continue
        src_path = _tile_src(row)
        remote = src_path.startswith(("/vsis3", "/vsicurl"))
        stem = Path(row["path"]).stem[:14]
        vals = None
        for attempt in range(MAX_TILE_ATTEMPTS):    # reopen through transient/fluctuating network
            try:
                with rasterio.open(src_path) as src:
                    tx, ty = Transformer.from_crs(target_epsg, src.crs, always_xy=True).transform(xs[gi], ys[gi])
                    vals = np.array(list(src.sample(zip(np.asarray(tx), np.asarray(ty)))), dtype=np.float32)
                break
            except (rasterio.errors.RasterioIOError, RuntimeError) as e:
                if attempt == MAX_TILE_ATTEMPTS - 1:
                    print(f"    {row['utm_zone']} {stem}: read failed after {MAX_TILE_ATTEMPTS} attempts ({e}); skipping tile")
                    break
                if remote:
                    # source.coop sits behind Cloudflare and throttles bursts of range
                    # requests (HTTP 520/429/5xx). Retrying instantly makes it worse, so
                    # back off exponentially with jitter; only wait on the link when the
                    # host is genuinely unreachable (DNS/connection down).
                    if not _online():
                        _wait_online()
                    else:
                        delay = min(BACKOFF_CAP, BACKOFF_BASE * (2 ** attempt))
                        delay *= 0.7 + 0.6 * random.random()      # jitter, avoid lockstep
                        print(f"    retry {attempt+1}/{MAX_TILE_ATTEMPTS} on {stem} "
                              f"({type(e).__name__}); backing off {delay:.0f}s", flush=True)
                        time.sleep(delay)
                        continue
                print(f"    retry {attempt+1}/{MAX_TILE_ATTEMPTS} on {stem} ({type(e).__name__})", flush=True)
                if not remote:
                    time.sleep(2)
        if vals is None:
            continue
        cover = np.ptp(vals, axis=1) > 0     # a point outside the raster -> constant nodata
        AE[gi[cover]] = vals[cover]
        assigned[gi[cover]] = True
        print(f"    {row['utm_zone']} {Path(row['path']).stem[:14]}: {int(cover.sum())}/{gi.size} points", flush=True)
        if remote:
            time.sleep(TILE_PAUSE)           # stay under the bucket's rate limit

    valid = ~np.isnan(AE).all(axis=1)
    out = df.reset_index(drop=True).copy()
    for d in range(AE_DIM):
        out[f"ae_{d}"] = AE[:, d]
    out = out[valid].reset_index(drop=True)
    frac = len(out) / n
    print(f"  AE sampled for {len(out):,}/{n:,} points ({n-len(out):,} no coverage, {frac:.0%} covered)")
    if frac < 0.5:
        raise RuntimeError(
            f"Only {frac:.0%} of points got AE coverage ({len(out):,}/{n:,}). This almost "
            f"always means the /vsicurl stream to data.source.coop was dropped mid-run "
            f"(check for 'Could not resolve host' / read errors above), not genuine "
            f"no-coverage. Re-run on a stable/fast network (the cluster) rather than "
            f"saving a partial cells.parquet.")
    return out


def download_ae_tiles(df: pd.DataFrame, year: int, target_epsg: int,
                      zones: tuple | None = None, scale: int = 10) -> int:
    """Fetch the AE tiles covering df's points, average-pool 10m->100m, and save each as
    a local <stem>_100m.tiff under AE_RAW_DIR/<year>/<zone>/ (exactly the layout Germany
    uses). sample_ae_at_points then reads these LOCAL files -- no per-point network reads,
    so sampling thousands of precincts is fast instead of taking hours.

    This is the "Germany way", done efficiently: it reads via /vsis3 (direct S3, bypassing
    the Cloudflare throttle/UA block) at a decimated out_shape, so GDAL pulls the tile's
    100m OVERVIEW (~tens of MB) rather than the full ~800MB raster. Only tiles that contain
    >=1 point are fetched. Safe to re-run: existing _100m.tiff files are skipped.
    Returns the number of tiles now present locally.
    """
    import os, socket, time
    from pyproj import Transformer
    from rasterio.enums import Resampling

    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    os.environ.setdefault("AWS_REGION", "us-west-2")

    def _online(host="s3.us-west-2.amazonaws.com", port=443, timeout=5) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    idx = pd.read_parquet(AE_RAW_DIR / "aef_index.parquet")

    # If df carries polygons (GeoDataFrame), select tiles by block-BBOX OVERLAP so every
    # tile a block touches is fetched -> the later zonal mean is exact (nothing skipped).
    # Otherwise (points, e.g. voting precincts) select tiles that CONTAIN a point.
    # GLOBAL extent from the interior-point centroids (x_mp/y_mp are guaranteed finite by the
    # loader). Polygon bounds are NOT used for the extent: TIGER has a few empty/null-geometry
    # blocks whose reprojected bounds are NaN, and a single NaN would poison a plain min()/max()
    # -> zero tiles. Per-tile selection below still uses polygon overlap where geometry is valid.
    xs = df["x_mp"].values.astype(float); ys = df["y_mp"].values.astype(float)
    clon, clat = Transformer.from_crs(target_epsg, 4326, always_xy=True).transform(xs, ys)
    clon = np.asarray(clon); clat = np.asarray(clat)
    cfin = np.isfinite(clon) & np.isfinite(clat)
    if not cfin.any():
        raise ValueError("no finite centroid coordinates in df -- cannot locate AE tiles.")
    loX, hiX = float(np.nanmin(clon[cfin])), float(np.nanmax(clon[cfin]))
    loY, hiY = float(np.nanmin(clat[cfin])), float(np.nanmax(clat[cfin]))

    has_geom = hasattr(df, "geometry") and getattr(df, "geometry", None) is not None
    if has_geom:
        b = df.geometry.to_crs(4326).bounds          # per-block wgs84 bbox (may contain NaN)
        bminx = b["minx"].values; bmaxx = b["maxx"].values
        bminy = b["miny"].values; bmaxy = b["maxy"].values

    print(f"  [ae-download] region extent (wgs84): lon [{loX:.3f}, {hiX:.3f}]  "
          f"lat [{loY:.3f}, {hiY:.3f}]  has_geom={has_geom}  "
          f"crs={getattr(getattr(df, 'geometry', None), 'crs', None)}", flush=True)
    m = (idx.year == year)
    n_year = int(m.sum())
    if zones is not None:
        m &= idx.utm_zone.isin(list(zones))
    n_zone = int(m.sum())
    m &= ((idx.wgs84_east >= loX) & (idx.wgs84_west <= hiX) &
          (idx.wgs84_north >= loY) & (idx.wgs84_south <= hiY))
    hits = idx[m].reset_index(drop=True)
    print(f"  [ae-download] index filter: year={n_year} -> +zones={n_zone} -> +bbox={len(hits)}", flush=True)

    # A tile is NEEDED if it CONTAINS a block centroid (always valid) OR its bbox overlaps a
    # block polygon (valid geometry only; NaN bounds compare False, harmless). The centroid
    # term guarantees every block's own tile is fetched even if its polygon is broken.
    need = []
    for _, row in hits.iterrows():
        w, e, s, nth = row["wgs84_west"], row["wgs84_east"], row["wgs84_south"], row["wgs84_north"]
        want = (clon >= w) & (clon <= e) & (clat >= s) & (clat <= nth)
        if has_geom:
            want = want | ((bmaxx >= w) & (bminx <= e) & (bmaxy >= s) & (bminy <= nth))
        if want.any():
            need.append(row)
    print(f"  [ae-download] {len(need)} tiles {'overlap the polygons' if has_geom else 'cover the points'} "
          f"(of {len(hits)} in bbox); resampling 10m->100m into {AE_RAW_DIR}/{year}/ ...", flush=True)
    if not need:
        raise FileNotFoundError(
            f"No AE tiles indexed for year={year} over this region (index has years "
            f"{sorted(idx.year.unique())}). Check aef_index.parquet covers year {year}.")

    # How many are already local vs still to fetch. If anything must be downloaded, we need
    # network -- probe once and fail LOUDLY here (rather than half-building an empty parquet).
    todo = [row for row in need
            if not (AE_RAW_DIR / str(year) / str(row["utm_zone"]) /
                    f"{Path(row['path']).stem}_100m.tiff").exists()]
    if todo and not _online():
        raise ConnectionError(
            f"{len(todo)} AE {year} tiles must be downloaded but S3 (s3.us-west-2.amazonaws.com) "
            f"is unreachable. Run the build on a node WITH internet so the tiles can be fetched "
            f"(they are then cached locally and later runs need no network).")

    MAX_ATTEMPTS = 6
    got = have = 0
    for i, row in enumerate(need, 1):
        zone = str(row["utm_zone"]); stem = Path(row["path"]).stem
        dest = AE_RAW_DIR / str(year) / zone / f"{stem}_100m.tiff"
        if dest.exists():
            have += 1
            continue
        src_url = str(row["path"]).replace("s3://", "/vsis3/", 1)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                with rasterio.open(src_url) as src:
                    nh, nw = max(1, src.height // scale), max(1, src.width // scale)
                    data = src.read(out_shape=(src.count, nh, nw), resampling=Resampling.average)
                    transform = src.transform * src.transform.scale(src.width / nw, src.height / nh)
                    profile = src.profile.copy()
                    profile.update(height=nh, width=nw, transform=transform,
                                   dtype="float32", compress="lzw", nodata=-128.0)
                break
            except Exception as e:                       # transient S3/throttle/fluctuation
                if attempt == MAX_ATTEMPTS:
                    raise RuntimeError(f"failed to read {src_url} after {MAX_ATTEMPTS} tries: {e}")
                delay = min(60, 2 ** attempt)
                print(f"    [{i}/{len(need)}] {zone}/{stem}: retry {attempt}/{MAX_ATTEMPTS} "
                      f"in {delay}s ({type(e).__name__})", flush=True)
                if not _online():
                    while not _online():
                        print("      offline; waiting for S3 ...", flush=True); time.sleep(10)
                else:
                    time.sleep(delay)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part.tiff")
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(data.astype("float32"))
        os.replace(tmp, dest)
        got += 1
        print(f"    [{i}/{len(need)}] {zone}/{dest.name}  {dest.stat().st_size/1e6:.0f} MB", flush=True)

    ready = got + have
    print(f"  [ae-download] {got} resampled, {have} already local ({ready}/{len(need)} ready)", flush=True)
    if ready < len(need):
        raise RuntimeError(f"only {ready}/{len(need)} AE {year} tiles are ready -- download incomplete.")
    return ready


def sample_ae_zonal_mean(gdf, year: int, target_epsg: int, zones: tuple | None = None):
    """Area-pooled AE per polygon: the MEAN AE over every 100m cell inside each polygon
    (zonal mean), read from the LOCAL 100m tiles. This is the correct analog of the
    Germany grid (where each unit IS a 100m cell) -- a single centroid sample is noisy
    for large/irregular units, so we average all covered cells instead.

    gdf: a GeoDataFrame with 'geometry', 'unit_id', 'x_mp'/'y_mp' (target_epsg, used for a
    fallback), and any label columns to carry through. Polygons with no interior cell
    centre (smaller than a 100m cell) fall back to the single containing cell via a point
    sample. Returns a plain DataFrame = carried columns + ae_0..63, dropping no-coverage.
    """
    import os
    import rasterio.features
    from rasterio.transform import rowcol
    from pyproj import Transformer

    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    os.environ.setdefault("AWS_REGION", "us-west-2")

    idx = pd.read_parquet(AE_RAW_DIR / "aef_index.parquet")
    gdf = gdf.reset_index(drop=True)
    n = len(gdf)

    # centroid lon/lat + x/y in target_epsg -> assign each block to ONE UTM zone, so we
    # reproject/index each block once (to its own zone) instead of all blocks to all zones.
    xs = gdf["x_mp"].values.astype(float); ys = gdf["y_mp"].values.astype(float)
    lon, lat = Transformer.from_crs(target_epsg, 4326, always_xy=True).transform(xs, ys)
    lon = np.asarray(lon); lat = np.asarray(lat)
    znum = np.clip(np.floor((lon + 180.0) / 6.0).astype(int) + 1, 1, 60)
    block_zone = np.array([f"{z}N" for z in znum])

    m = (idx.year == year)
    if zones is not None:
        m &= idx.utm_zone.isin(list(zones))
    m &= ((idx.wgs84_east >= lon.min()) & (idx.wgs84_west <= lon.max()) &
          (idx.wgs84_north >= lat.min()) & (idx.wgs84_south <= lat.max()))
    hits = idx[m].reset_index(drop=True)
    if hits.empty:
        raise FileNotFoundError(f"No AE tiles for year={year} covering the polygons.")

    sum_ae = np.zeros((n, AE_DIM), dtype=np.float64)   # zonal-mean accumulators
    cnt    = np.zeros(n, dtype=np.int64)
    fb_ae  = np.full((n, AE_DIM), np.nan, dtype=np.float32)   # centroid fallback (sub-cell blocks)

    def _local_tile(row):
        # LOCAL only: streaming a missing tile over /vsis3 costs ~90s each and the zonal
        # loop touches many edge tiles. Every block's centroid tile is guaranteed local
        # (the prefetch downloads point-containing tiles), so blocks keep their coverage.
        stem = Path(row["path"]).stem
        for name in (f"{stem}_100m.tiff", f"{stem}.tiff"):
            p = AE_RAW_DIR / str(year) / str(row["utm_zone"]) / name
            if p.exists():
                return str(p)
        return None

    zlist = sorted(hits["utm_zone"].unique())
    skipped = 0
    print(f"  [zonal] pooling AE over {n:,} polygons, {len(hits)} tiles, {len(zlist)} zones ...", flush=True)
    for zi, zone in enumerate(zlist, 1):
        zpos   = np.where(block_zone == zone)[0]
        ztiles = hits[hits["utm_zone"] == zone]
        if zpos.size == 0 or ztiles.empty:
            continue
        zone_crs = None
        for _, r0 in ztiles.iterrows():
            p0 = _local_tile(r0)
            if p0 is not None:
                with rasterio.open(p0) as s0:
                    zone_crs = s0.crs
                break
        if zone_crs is None:
            continue
        gz     = gdf.iloc[zpos].to_crs(zone_crs)          # reproject THIS zone's blocks once
        geoms  = gz.geometry.values
        sindex = gz.sindex                                # index over the zone subset only
        czx, czy = Transformer.from_crs(target_epsg, zone_crs, always_xy=True).transform(xs[zpos], ys[zpos])
        czx = np.asarray(czx); czy = np.asarray(czy)
        print(f"    zone {zone} [{zi}/{len(zlist)}]: {zpos.size:,} blocks, {len(ztiles)} tiles", flush=True)

        for _, row in ztiles.iterrows():
            tpath = _local_tile(row)
            if tpath is None:
                skipped += 1
                continue
            with rasterio.open(tpath) as src:
                ae = src.read()                            # (64, H, W)
                nodata = src.nodata
                tb = src.bounds
                cand = np.asarray(list(sindex.intersection((tb.left, tb.bottom, tb.right, tb.top))))
                if cand.size == 0:
                    continue
                shapes = ((geoms[c], int(zpos[c]) + 1) for c in cand
                          if geoms[c] is not None and not geoms[c].is_empty)
                blk = rasterio.features.rasterize(
                    shapes, out_shape=(src.height, src.width), transform=src.transform,
                    fill=0, all_touched=True, dtype="int32")
                nod = (ae[0] == nodata) if nodata is not None else np.zeros(ae.shape[1:], bool)
                valid = (blk > 0) & ~nod
                if valid.any():
                    ids = blk[valid] - 1
                    aef = ae[:, valid].T.astype(np.float64)          # (nvalid, 64)
                    # grouped sum via bincount over the tile's blocks (fast; np.add.at is not)
                    uniq, inv = np.unique(ids, return_inverse=True)
                    cnt[uniq] += np.bincount(inv, minlength=uniq.size)
                    for d in range(AE_DIM):
                        sum_ae[uniq, d] += np.bincount(inv, weights=aef[:, d], minlength=uniq.size)
                # centroid fallback for the same candidates (fills sub-cell blocks cheaply)
                rr, cc = rowcol(src.transform, czx[cand], czy[cand], op=np.floor)
                rr = np.asarray(rr).astype(np.int64); cc = np.asarray(cc).astype(np.int64)
                inb = (rr >= 0) & (rr < src.height) & (cc >= 0) & (cc < src.width)
                if inb.any():
                    gi = zpos[cand[inb]]
                    vals = ae[:, rr[inb], cc[inb]].T.astype(np.float32)   # (k, 64)
                    ok = ~(vals[:, 0] == nodata) if nodata is not None else np.ones(len(vals), bool)
                    fb_ae[gi[ok]] = vals[ok]
        del gz, geoms, sindex

    mean_ae = np.full((n, AE_DIM), np.nan, dtype=np.float32)
    has = cnt > 0
    mean_ae[has] = (sum_ae[has] / cnt[has, None]).astype(np.float32)
    miss = ~has & ~np.isnan(fb_ae).all(axis=1)            # sub-cell blocks -> centroid value
    mean_ae[miss] = fb_ae[miss]
    print(f"  [zonal] {int(has.sum()):,} zonal-mean + {int(miss.sum()):,} centroid-fallback / {n:,}"
          f"  ({skipped} non-local tiles skipped)", flush=True)

    out = gdf.drop(columns="geometry").reset_index(drop=True)
    for d in range(AE_DIM):
        out[f"ae_{d}"] = mean_ae[:, d]
    valid_row = ~np.isnan(mean_ae).all(axis=1)
    cov = valid_row.mean() if n else 0.0
    if cov < 0.5:
        raise RuntimeError(
            f"zonal AE coverage {cov:.1%} ({int(valid_row.sum()):,}/{n:,}) -- almost no local "
            f"tiles for year={year} ({skipped} skipped as non-local). The tiles were never "
            f"downloaded: run on an internet node so download_ae_tiles can fetch year {year}, "
            f"or set the build to a year whose *_100m.tiff tiles already exist.")
    out = out[valid_row].reset_index(drop=True)
    print(f"  [zonal] AE pooled for {len(out):,}/{n:,} polygons ({n-len(out):,} no coverage)", flush=True)
    return out


# ============================================================
# 3. Build full aligned dataset
# ============================================================

def build_dataset(
    region: str = "berlin",
    year: int = 2022,
    normalize: str = "R0",
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Join AE features with all Zensus variables -> flat DataFrame.

    Saves to data/processed/<region>_100m_features_labels.parquet
    and a sidecar .meta.json.

    Columns: grid_id, x_mp, y_mp, ae_0…ae_63, population,
             rent_per_m2, avg_age, vacancy_rate, foreigner_share, citizens_18plus

    Parameters
    ----------
    region    : key in REGION_BBOX or custom
    year      : AEF vintage
    normalize : R0 / R1 / R2 for AE embeddings
    variables : which Zensus variables to load (default: all available)
    """
    if variables is None:
        variables = ["population", "rent_per_m2", "avg_age",
                     "vacancy_rate", "foreigner_share", "citizens_18plus"]

    bbox = REGION_BBOX[region]
    print(f"[build_dataset] region={region} year={year} normalize={normalize}")
    print(f"  EPSG:3035 bbox: {bbox}")

    # --- Load population first (defines the cell universe) ---
    print("  Loading population ...")
    pop_df = load_zensus("population", bbox)
    print(f"  {len(pop_df):,} populated cells in region")

    # --- Sample AE embeddings at those centroids ---
    print("  Sampling AE embeddings ...")
    df = sample_ae_at_zensus_cells(pop_df, year=year, normalize=normalize)
    print(f"  {len(df):,} cells with AE coverage")

    # --- Join additional Zensus variables ---
    for var in variables:
        if var == "population":
            continue
        if not (ZENSUS_RAW_DIR / ZENSUS_META[var]["file"]).exists():
            print(f"  Skipping {var} (not downloaded)")
            continue
        print(f"  Loading {var} ...")
        var_df = load_zensus(var, bbox)[["grid_id", var]]
        df = df.merge(var_df, on="grid_id", how="left")

    # --- Save ---
    region_dir = DATA_PROCESSED / region
    region_dir.mkdir(parents=True, exist_ok=True)
    out_path = region_dir / "cells.parquet"
    df.to_parquet(out_path, index=False)

    print(f"  Saved -> {out_path}  ({len(df):,} rows, {len(df.columns)} cols)")
    return df


def build_dataset_allcells(
    base_region: str,
    out_region: str | None = None,
    year: int = 2022,
    boundary_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build an ALL-CELLS dataset: every 100m land cell in the region's bbox, with
    AE embeddings, labelled with the Zensus population where present and **0**
    otherwise (uninhabited land -> pop 0). This lets the model learn the pop≈0 AE
    signature (forest/water) instead of only ranking known-inhabited cells.

    IMPORTANT (borders): absent -> 0 is only valid INSIDE the census's country. For
    Berlin the bbox lies entirely within Germany, so no clip is needed. For a bbox
    that crosses a national border (e.g. Germany, whose bbox overlaps FR/PL/CZ/...),
    pass `boundary_path` to a national-border polygon: cells OUTSIDE the border are
    DROPPED (never labelled 0), so the model is not taught that foreign towns are
    empty. Cells inside the border but absent from the census -> 0 (uninhabited).

    Saves to data/processed/<out_region>/cells.parquet (out_region defaults to
    "<base_region>_all"). Columns match build_dataset (grid_id, x_mp, y_mp, ae_*,
    population) so the runner/aggregation work unchanged.
    """
    out_region = out_region or f"{base_region}_all"
    bbox = REGION_BBOX[base_region]
    west, south, east, north = bbox
    print(f"[build_dataset_allcells] base={base_region} -> {out_region}  bbox={bbox}")

    # Inhabited cells (defines the 100m lattice offset + the labels)
    pop = load_zensus("population", bbox)
    ox = int(pop.x_mp.iloc[0]) % 100
    oy = int(pop.y_mp.iloc[0]) % 100

    # Full 100m grid over the bbox, aligned to the Zensus lattice
    xs = np.arange((west // 100) * 100 + ox, east + 1, 100)
    ys = np.arange((south // 100) * 100 + oy, north + 1, 100)
    xs = xs[(xs >= west) & (xs <= east)]
    ys = ys[(ys >= south) & (ys <= north)]
    XX, YY = np.meshgrid(xs, ys)
    xr = XX.ravel().astype("int32")
    yr = YY.ravel().astype("int32")
    print(f"  full grid: {len(xr):,} cells ({len(xs)}x{len(ys)})")

    # Boundary mask: keep only cells INSIDE the national border. Foreign cells are
    # DROPPED (not labelled 0). We rasterize the border onto the 100m grid once and
    # look up each cell's pixel -> O(cells), no per-point polygon test.
    if boundary_path is not None:
        gdf  = gpd.read_file(boundary_path).to_crs(EPSG_INSPIRE)
        geom = gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union
        transform = rasterio.transform.from_origin(float(xs.min()) - 50, float(ys.max()) + 50, 100, 100)
        inside = geometry_mask([geom], out_shape=(len(ys), len(xs)), transform=transform, invert=True)
        col  = ((xr - int(xs.min())) // 100).astype(np.int64)
        row  = ((int(ys.max()) - yr) // 100).astype(np.int64)
        keep = inside[row, col]
        xr, yr = xr[keep], yr[keep]
        print(f"  boundary mask ({Path(boundary_path).name}): kept "
              f"{int(keep.sum()):,}/{len(keep):,} cells inside the border "
              f"({100*keep.mean():.1f}%)")

    grid = pd.DataFrame({"x_mp": xr, "y_mp": yr})
    grid["grid_id"] = grid.x_mp.astype(str) + "_" + grid.y_mp.astype(str)

    # Sample AE at every grid cell (cells with no AE coverage are dropped)
    grid = sample_ae_at_zensus_cells(grid, year=year, normalize="R0")

    # Attach population: census value where present, 0 elsewhere (uninhabited)
    grid = grid.merge(pop[["x_mp", "y_mp", "population"]], on=["x_mp", "y_mp"], how="left")
    n_inhab = int(grid["population"].notna().sum())
    grid["population"] = grid["population"].fillna(0.0).astype("float32")

    if n_inhab != len(pop):
        print(f"  [WARN] {len(pop)-n_inhab} inhabited cells did not match the grid "
              f"(lattice misalignment or AE-dropped) - check before trusting labels")

    region_dir = DATA_PROCESSED / out_region
    region_dir.mkdir(parents=True, exist_ok=True)
    out_path = region_dir / "cells.parquet"
    grid.to_parquet(out_path, index=False)
    print(f"  Saved -> {out_path}  ({len(grid):,} cells; "
          f"{n_inhab:,} inhabited [{100*n_inhab/len(grid):.1f}%], "
          f"{len(grid)-n_inhab:,} zero; total pop {grid['population'].sum()/1e6:.3f}M)")
    return grid


# ============================================================
# 4. Build coarse blocks
# ============================================================

def build_coarse_blocks(
    fine_df: pd.DataFrame,
    region_m: int,
    target: str = "population",
) -> pd.DataFrame:
    """Aggregate fine cells into coarse blocks.

    Block ID = (x_block, y_block) from integer-dividing centroids by block size.

    Aggregation (Logic Module §1.3):
      counts  -> sum
      rates   -> population-weighted mean

    tau2_B = max(1, 0.01 * Y_B)  placeholder disclosure-control noise model.

    Parameters
    ----------
    fine_df : output of build_dataset or load from parquet
    region_m : region (coarse block) size in metres (200, 400, 1000, 4000, ...)
    target  : which Zensus variable is the label

    Returns
    -------
    DataFrame: block_id, n_cells, member_grid_ids, Y_<target>, tau2_B,
               mean_ae_0…mean_ae_63 (block-level mean AE for linear baseline)
    """
    block_size = region_m              # region (coarse block) size in metres
    is_count = target in COUNT_TARGETS

    if target not in fine_df.columns:
        return pd.DataFrame()

    # Assign each fine cell to its coarse block (integer coords -> avoids string keys)
    bx = ((fine_df["x_mp"].values // block_size) * block_size).astype(np.int64)
    by = ((fine_df["y_mp"].values // block_size) * block_size).astype(np.int64)
    ae_cols = [c for c in fine_df.columns if c.startswith("ae_")]

    d = pd.DataFrame({"bx": bx, "by": by, "t": fine_df[target].values})
    if not is_count:
        w = fine_df["population"].fillna(1).clip(lower=0).values if "population" in fine_df else np.ones(len(fine_df))
        d["w"] = w
        d["tw"] = d["t"].values * w
    for c in ae_cols:
        d[c] = fine_df[c].values
    d = d[np.isfinite(d["t"].values)]
    if d.empty:
        return pd.DataFrame()

    # Vectorized groupby (fast even at ~10^6 cells / ~10^5 blocks - no Python loop)
    g = d.groupby(["bx", "by"], sort=True)
    n_cells = g.size()
    if is_count:
        Y = g["t"].sum()
    else:
        wsum = g["w"].sum()
        Y = (g["tw"].sum() / wsum.replace(0, np.nan))
    agg = pd.DataFrame({"n_cells": n_cells, f"Y_{target}": Y}).reset_index()
    if ae_cols:
        mean_ae = g[ae_cols].mean().reset_index(drop=True)
        mean_ae.columns = [f"mean_ae_{i}" for i in range(len(ae_cols))]
        agg = pd.concat([agg, mean_ae], axis=1)

    agg = agg[np.isfinite(agg[f"Y_{target}"].values)].reset_index(drop=True)
    agg["block_x"] = agg["bx"].astype(int)
    agg["block_y"] = agg["by"].astype(int)
    agg["block_id"] = agg["bx"].astype(str) + "_" + agg["by"].astype(str)
    agg["tau2_B"] = np.maximum(1.0, 0.01 * agg[f"Y_{target}"].values)
    agg = agg.drop(columns=["bx", "by"])

    print(f"  {len(agg):,} blocks at region={region_m}m")
    return agg


# ============================================================
# 4b. Aggregate 100m cells -> subregions (the prediction/eval unit)
# ============================================================

def aggregate_to_subregions(cells: pd.DataFrame, subregion_m: int) -> pd.DataFrame:
    """Pool 100m cells into subregion units of size subregion_m metres.

    The subregion becomes the fine unit the model predicts and is evaluated on
    (a "region" is then split into these subregions). Ground
    truth population is SUMMED; AEF embeddings are MEAN-pooled; the subregion
    keeps a centroid on the subregion_m grid. subregion_m<=100
    returns the 100m cells unchanged.

    Population-only (counts). Rate targets are not supported with subregion>1.
    """
    if subregion_m <= FINE_RES_M:
        return cells
    s = int(subregion_m)
    ae_cols = [c for c in cells.columns if c.startswith("ae_")]
    df = cells.copy()
    df["_sx"] = (df["x_mp"] // s) * s
    df["_sy"] = (df["y_mp"] // s) * s
    agg = {c: "mean" for c in ae_cols}
    agg["population"] = "sum"
    out = df.groupby(["_sx", "_sy"], sort=False).agg(agg).reset_index()
    out["x_mp"]    = (out["_sx"] + s // 2).astype("int32")
    out["y_mp"]    = (out["_sy"] + s // 2).astype("int32")
    out["grid_id"] = out["_sx"].astype("int64").astype(str) + "_" + out["_sy"].astype("int64").astype(str)
    out = out.drop(columns=["_sx", "_sy"])
    print(f"  aggregated {len(cells):,} 100m cells -> {len(out):,} subregions of {s}m")
    return out


# ============================================================
# 5. Spatial CV splits
# ============================================================

def make_spatial_splits(
    block_df: pd.DataFrame,
    scheme: str = "berlin_5fold",
    seed: int = 42,
    n_folds: int = 5,
    super_block_size: int = 5000,
) -> dict[int, list[str]]:
    """Assign each coarse block to a CV fold and save as JSON.

    berlin_5fold:
      Tile the region into 5km × 5km super-blocks, assign super-blocks
      randomly to folds, then assign each coarse block to the fold of its
      super-block centroid. This keeps spatially contiguous regions together.

    Hard rules:
      - Every block belongs to exactly one fold (centroid rule).
      - Split JSON is saved to data/splits/ and loaded from disk; never
        regenerated at runtime (Logic Module §4, Rule 10).

    Returns {fold_id: [block_ids]}
    """
    rng = np.random.default_rng(seed)

    # Super-block assignment via centroid
    block_df = block_df.copy()
    block_df["super_x"] = (block_df["block_x"] // super_block_size) * super_block_size
    block_df["super_y"] = (block_df["block_y"] // super_block_size) * super_block_size
    block_df["super_id"] = (block_df["super_x"].astype(str) + "_"
                            + block_df["super_y"].astype(str))

    super_ids = block_df["super_id"].unique()
    perm = rng.permutation(len(super_ids))
    fold_of_super = {sid: int(perm[i] % n_folds)
                     for i, sid in enumerate(super_ids)}
    block_df["fold"] = block_df["super_id"].map(fold_of_super)

    splits: dict[int, list[str]] = {f: [] for f in range(n_folds)}
    for _, row in block_df.iterrows():
        splits[int(row["fold"])].append(str(row["block_id"]))

    # Save
    splits_dir = DATA_PROCESSED.parent / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    out_path = splits_dir / f"{scheme}_seed{seed}.json"
    with out_path.open("w") as f:
        json.dump(splits, f, indent=2)

    sizes = {k: len(v) for k, v in splits.items()}
    print(f"  Splits saved -> {out_path}")
    print(f"  Fold sizes: {sizes}")
    return splits


def load_splits(path: str | Path) -> dict[int, list[str]]:
    """Load a saved split JSON. Keys are ints."""
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ============================================================
# 6. PyTorch Dataset
# ============================================================

class CoarseBlockDataset(Dataset):
    """One coarse block per item: (x_fine, y_coarse, tau2_B, block_id).

    Block sizes vary at region boundaries; use collate_variable_blocks
    as the DataLoader collate_fn.
    """

    def __init__(
        self,
        fine_df: pd.DataFrame,
        block_df: pd.DataFrame,
        target: str,
        block_list: list[str],
    ) -> None:
        ae_cols = [c for c in fine_df.columns if c.startswith("ae_")]
        self.x_fine    = torch.tensor(fine_df[ae_cols].values,  dtype=torch.float32)
        self.fine_gids = fine_df["grid_id"].values
        self.target    = target

        # Index fine cells by block
        self._block_data: list[tuple[Tensor, float, float, str]] = []
        gid_to_idx = {gid: i for i, gid in enumerate(self.fine_gids)}

        for bid in block_list:
            row = block_df[block_df["block_id"] == bid]
            if row.empty:
                continue
            row = row.iloc[0]
            member_gids = row["member_grid_ids"]
            fine_idx = [gid_to_idx[g] for g in member_gids if g in gid_to_idx]
            if not fine_idx:
                continue
            x_block  = self.x_fine[fine_idx]
            y_block  = float(row[f"Y_{target}"])
            tau2_b   = float(row["tau2_B"])
            self._block_data.append((x_block, y_block, tau2_b, bid))

    def __len__(self) -> int:
        return len(self._block_data)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, Tensor, str]:
        x, y, tau2, bid = self._block_data[i]
        return x, torch.tensor(y, dtype=torch.float32), \
               torch.tensor(tau2, dtype=torch.float32), bid


def collate_variable_blocks(
    batch: list[tuple[Tensor, Tensor, Tensor, str]],
) -> tuple[list[Tensor], Tensor, Tensor, list[str]]:
    """Collate blocks with variable numbers of fine cells."""
    xs, ys, taus, bids = zip(*batch)
    return (
        list(xs),
        torch.stack(list(ys)),
        torch.stack(list(taus)),
        list(bids),
    )