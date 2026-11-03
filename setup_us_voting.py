"""
Puffin data setup -- US (NYT 2024 presidential precinct voting).

Builds data/processed/us_voting/cells.parquet: one row per precinct, with 64-dim
AlphaEarth features sampled at the precinct's interior point, plus vote counts
(votes_rep / votes_dem / votes_total / votes_other) and the coarse unit ids
(county_fips, state_fips) used as the disaggregation region.

Unlike Germany this is a FIXED-region dataset: the fine unit is the precinct
(a real polygon, not a grid cell), so there is no region x subregion size sweep --
only the coarse level varies (county -> precinct, state -> precinct).

AlphaEarth tiles are STREAMED from the public source.coop bucket over HTTPS
(no local tiles, no AWS credentials). This needs working outbound access to
data.source.coop -- if DNS/network there fails the build stops with a clear error
rather than saving a partial file.

Edit the CONFIG block below, then run:
    python setup_us_voting.py

Then:
    python main_us.py   # DATASET='us_voting'

For German population data use setup_germany.py instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

# =============================================================================
# CONFIG -- edit these values, then run: python setup_us_voting.py
# =============================================================================
STATES = None      # which states to build:
                     #   None          -> ENTIRE US (all precincts nationwide)
                     #   ["PA"]        -> a single state
                     #   ["PA","MI"]   -> several states
                     # Use two-letter postal codes. Start with one state to verify
                     # the pipeline before committing to a nationwide build.

YEAR = 2024          # election year (NYT precinct map year)

PREFETCH_TILES = True   # True (recommended): fetch each covering tile's 100m OVERVIEW from
                        # S3 once (~tens of MB/tile), average-pool to a local 100m COG, then
                        # sample from local files. Sampling thousands of precincts by streaming
                        # per-point from S3 takes HOURS (per-point network latency); local
                        # sampling is fast. This also matches Germany's 100m-pooled features.
                        # False = stream per-point directly (only ok for a handful of points).

POOL = "zonal"          # "zonal": AE per precinct = MEAN over all 100m cells in the precinct
                        #          polygon, plus exposure = precinct area (same as census blocks).
                        # "point": single AE sample at the precinct interior point, no area (v1).

OUT_REGION = "us_voting"   # -> data/processed/<OUT_REGION>/cells.parquet
# =============================================================================


def main() -> None:
    from puffin.datasets.us_voting import build_us_voting

    scope = "ENTIRE US" if STATES is None else ", ".join(STATES)
    print(f"\n{'='*60}")
    print(f"  Puffin setup [US VOTING]  states={scope}  year={YEAR}")
    print(f"{'='*60}\n")

    build_us_voting(year=YEAR, states=STATES, out_region=OUT_REGION, prefetch=PREFETCH_TILES,
                    pool=POOL)

    print(f"\n{'='*60}")
    print(f"  Setup complete.")
    print(f"  data/processed/{OUT_REGION}/cells.parquet is ready.")
    print(f"  Run: python main_us.py   # DATASET='us_voting'")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
