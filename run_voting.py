"""
US voting disaggregation, per party: county total -> precinct, with the Gaussian Puffin method.

Runs the exact main_us.py pipeline (same folds, training, split, calibration, scoring) and
only overrides its configuration, so main_us.py stays set up for the population runs.
Precincts are irregular polygons, so -- exactly as for the census blocks -- the embedding is the
mean over each precinct polygon and the area term is used (exposure = precinct area).

    python run_voting.py
Output: experiments/us_voting_gaussian_summary.json (one row per party, 5-seed mean +/- std).
"""
import main_us as M

M.DATASET      = "us_voting"
M.TARGETS      = ["votes_rep", "votes_dem", "votes_other", "votes_total"]
M.LEVELS       = {"county": "county_fips"}
M.EXPOSURE_COL = "exposure"
M.METHOD       = "gaussian"
M.SEEDS        = [0, 1, 2, 3, 4]
M.SAVE_PREDS   = True

if __name__ == "__main__":
    M.main()
