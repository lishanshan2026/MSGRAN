# MS-GRAN Code

This repository contains the core source code and compact benchmark metadata for MS-GRAN experiments on multi-station GNSS displacement-residual forecasting.

The code covers data acquisition, preprocessing, graph construction, baseline training, residual-adaptation training, control experiments, sensitivity analyses, and evaluation utilities. Large processed arrays, generated figures, model checkpoints, and downstream document files are not included.

## Layout

```text
src/              Core Python scripts
metadata/         Station identifiers, fixed split, and Top-K adjacency files
requirements.txt  Python package requirements
README.md         Repository overview
```

## Benchmark metadata

The `metadata/` directory provides the station order and identifiers for the 90-station Cascadia benchmark, the fixed chronological split, and the training-only row-normalized Top-K adjacency matrix used by the main MS-GRAN configuration. These files are intended to make the benchmark protocol and leakage controls directly inspectable.

## Data

Raw daily GNSS position time series are publicly available from the Nevada Geodetic Laboratory. The benchmark uses processed NGL tenv3 products and constructs displacement residuals with training-period preprocessing parameters. Large intermediate arrays can be regenerated with the scripts in `src/` or provided by the authors upon reasonable request.

## Reproducibility outline

1. Download or prepare NGL daily GNSS time series.
2. Build the standardized displacement-residual benchmark.
3. Construct the training-only correlation-distance graph.
4. Train station-local baselines and residual-adaptation models.
5. Evaluate validation-selected checkpoints on the frozen test partition.

## Contact

Corresponding author: Qingjie Liu, liuqingjie@cidp.edu.cn
