"""Dataset builders. Each module turns a raw source into the standardized
data/processed/<name>/cells.parquet that the shared runner consumes.

Two schema flavours:
  * GRID  (e.g. germany population): x_mp, y_mp, grid_id, ae_0..63, <target>.
    Region/subregion are chosen at run time from the 100m grid (parametric).
  * FIXED  (e.g. us voting): unit_id, region_id, x_mp, y_mp, ae_0..63, <targets>.
    `region_id` is a fixed coarse unit (e.g. county); the row is the subregion.

The runner detects a `region_id` column and switches to fixed mode; otherwise it
uses the grid. The method (Poisson loss, conserving posterior, calibration,
within-region R2) is identical for both.
"""
