# Benchmark metadata

This directory contains compact metadata for reproducing the Cascadia 90-station benchmark setup used by MS-GRAN.

## Files

- `station_list_90.csv`: station order, station identifiers, coordinates, observed-day counts, and coverage fractions.
- `chronological_split.csv`: the fixed chronological training, validation, and frozen-test periods.
- `adjacency_matrix_top8.csv`: the row-normalized non-negative Top-K adjacency matrix used by the main MS-GRAN configuration.
- `adjacency_edges_top8.csv`: edge-list view of the same adjacency matrix, including the training-period correlation evidence and distance support used to form each retained edge.

## Split protocol

The benchmark uses a fixed chronological split: 2010-01-01 to 2019-12-31 for training, 2020-01-01 to 2021-12-31 for validation, and 2022-01-01 to 2024-12-31 as the frozen test period. Test observations are used only for final evaluation.

## Adjacency construction

The graph is constructed only from the 2010-2019 training partition. For each station pair, a 31-day causal EWMA high-frequency residual summary is computed, component-wise Pearson correlations are estimated over mutually observed training days, and the median signed correlation across E/N/U is used as pairwise correlation evidence. Candidate neighbours are ranked by absolute training correlation, with K = 8 directed candidates retained for each station. The retained candidate weights combine absolute training correlation and geographic-distance support with alpha = 0.75 for correlation and 0.25 for distance. Self-loops are included, the support is symmetrized by element-wise maximum, and each row is normalized to sum to one.

No validation or frozen-test observations are used to construct graph edges or graph weights.
