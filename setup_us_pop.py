"""
Puffin data setup -- US census-block population (2020).

Builds data/processed/us_population/cells.parquet: one row per census block, with 64-dim
AlphaEarth features sampled at the block's interior point (EPSG:5070) and its 2020
population (POP20). Coarse units (county / tract) are derived from the block GEOID.

Source: Census TIGER/Line 2020 TABBLOCK20 shapefiles (one zip per state; bundle geometry
+ population + interior point). AE tiles are the SAME ones voting uses -- for states you
already built for voting there is no new tile download, just the block labels.

Edit the CONFIG block below, then run:
    python setup_us_pop.py
Then:
    python main_us.py   # DATASET='us_population'
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

# =============================================================================
# CONFIG -- edit these values, then run: python setup_us_pop.py
# =============================================================================
STATES = None      # which states to build:
                     #   None          -> ENTIRE US (all census blocks; ~8M, cluster only)
                     #   ["PA"]        -> a single state (~420k blocks)
                     #   ["PA","OH"]   -> several states
                     # Use two-letter postal codes. Start with one state to verify.

PREFETCH_TILES = True   # True: download+resample covering AE tiles to local 100m COGs first,
                        # then sample OFFLINE (fast). Tiles are shared with us_voting, so a
                        # state already built for voting refetches nothing.

POOL = "zonal"          # "zonal": AE per block = MEAN over all 100m cells in the block
                        #          polygon (the correct grid analog; needs block geometry).
                        # "point": single AE sample at the block interior point (v1, noisy
                        #          for large/irregular blocks).

AE_YEAR = 2020          # AlphaEarth vintage. Blocks + population + housing are the 2020
                        # Census, so 2020 is the matched ground-truth vintage. The setup
                        # ensures the 2020 tiles are present (downloads any missing) before
                        # building. (Voting uses AE 2024 to match the 2024 election.)

OUT_REGION = "us_population"   # -> data/processed/<OUT_REGION>/cells.parquet
# =============================================================================


def main() -> None:
    from puffin.datasets.us_population import build_us_population

    scope = "ENTIRE US" if STATES is None else ", ".join(STATES)
    print(f"\n{'='*60}")
    print(f"  Puffin setup [US POPULATION]  states={scope}  AE {AE_YEAR}")
    print(f"{'='*60}\n")

    build_us_population(states=STATES, out_region=OUT_REGION, prefetch=PREFETCH_TILES,
                        pool=POOL, ae_year=AE_YEAR)

    print(f"\n{'='*60}")
    print(f"  Setup complete.")
    print(f"  data/processed/{OUT_REGION}/cells.parquet is ready.")
    print(f"  Run: python main_us.py   # DATASET='us_population'")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
