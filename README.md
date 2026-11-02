# Decision-focused Sparse Tangent Portfolio Optimization

Code for the paper *Decision-focused Sparse Tangent Portfolio Optimization*.

A predictive model (MLP) forecasts next-day asset returns from a rolling window of past
returns. The forecast is passed through a **differentiable OSCAR layer** — a DPP-compliant
convex reformulation of Sharpe-ratio maximization (CVXPYlayers) combined with a smooth
top-*k* selection operator — and the model is trained end-to-end on the realized portfolio
return under an exact cardinality constraint. The repository also contains the two
conventional pipelines used as baselines (Historic, PFL) and the three downstream
optimizers (OSCAR, SD-relaxation, mSSRM-PGA).

## Installation

Python 3.12 (tested) with the pinned versions in `requirements.txt`:

```bash
conda create -n sparse-dfl python=3.12
conda activate sparse-dfl
pip install -r requirements.txt
```

A CUDA-capable GPU is used by default (`--cuda 0`); pass `--cuda -1` to run on CPU.

## Data

Daily return matrices are generated from Yahoo Finance / FinanceDataReader and stored as
`data/daily/{market}_{YYMMDD}_{YYMMDD}.csv` (rows = dates, columns = assets, values =
simple returns). Data files are not distributed with the code.

```bash
python src/data_generator.py --data euro50    --start 2016-01-01 --end 2025-12-31
python src/data_generator.py --data ftse100   --start 2016-01-01 --end 2025-12-31
python src/data_generator.py --data nikkei225 --start 2016-01-01 --end 2025-12-31
python src/data_generator.py --data kospi200  --start 2016-01-01 --end 2025-12-31
```

`kospi200` needs the KRX constituent list saved as `data/raw/kospi200.csv` (CP949 encoded,
with a 6-digit code column named `종목코드`, as exported from the KRX data portal).
Constituents are taken as of the download date; the paper restricts attention to assets
with complete price histories over the sample period, which is what the generator's
`dropna(axis="columns")` does.

The default file suffix used by every script is `160101_251231`; override with
`--data_suffix` if you generate a different period.

Note that index constituents and Yahoo's adjusted prices change over time, so data
downloaded today differs from the snapshot used in the paper and the reported numbers
will not be reproduced exactly.

## Usage

All commands are run from the repository root. One invocation trains/evaluates one
random seed for every cardinality level in `--k_list` (given as a percentage of the number
of assets; `k = round(rho * N)`).

```bash
# DFL (ours): train through the differentiable OSCAR layer
python src/main.py --mode dfl --dataset euro50 --alpha 0.5 --seed 0

# PFL: train (or load) an MSE predictor, then evaluate it with each optimizer
python src/main.py --mode pfl --dataset euro50 --seed 0 --selector oscar ssrm sdr

# Historic: no training; the window mean of past returns is used as the expected return
python src/main.py --mode historic --dataset euro50 --selector oscar ssrm sdr

# Aggregate every run under artifacts/ into mean ± std over seeds
python src/aggregate.py --artifacts_dir artifacts
```


| Option | Meaning |
|---|---|
| `--mode {dfl,pfl,historic}` | training method (see below) |
| `--selector {oscar,sdr,ssrm,oscar_hard}` | downstream optimizer(s); `dfl` supports `oscar` only |
| `--k_list 10 15 20` | cardinality levels in % of assets |
| `--seed N` | random seed (ignored by `historic`) |
| `--alpha 0.5` | weight of the decision loss in `α·L_DFL + (1−α)·L_MSE` (`dfl`) |
| `--cov_mode {rolling,fixed}` | per-window covariance (default) or one covariance from the train split |
| `--sdr_stride N` | SD-relaxation only: solve every N steps and interpolate (it solves one SDP per step) |
| `--splits train test` | which splits to evaluate |
| `--save_dir artifacts` | output root; existing runs are skipped unless `--overwrite` |
| `--retrain`, `--model_dir` | `pfl`: force retraining / where the MSE checkpoint lives |

`--selector oscar` is the soft (differentiable) OSCAR layer run without gradients;
`oscar_hard` is the original three-stage heuristic with a hard top-*k* selection.

## Pipeline

| File | Role |
|---|---|
| `src/main.py` | argument parsing and the three experiment modes |
| `src/problem.py` | `SparseSharpeProblem`: data splits, `get_decision` (prediction → weights for a given optimizer), `get_objective` (`−rᵀw`) |
| `src/data.py` | CSV loading, standardization, chronological 80/20 split, 100-day windows, rolling covariance with shrinkage + jitter |
| `src/oscar.py` | `QCQPlayer` (Eq. 5) and `DiffbleOSCAR`: optimize → `\|L_Σᵀ w\|` → soft top-*k* → re-optimize |
| `src/selector.py` | differentiable top-*k* operator (bisection forward, closed-form vector–Jacobian backward) |
| `src/losses.py` | `mse` (PFL) and `dfl` (`α·L_DFL + (1−α)·L_MSE`, Algorithm 1) losses |
| `src/train.py` | training loop: one mini-batch per epoch, per-sample losses through the decision layer |
| `src/evaluate.py` | walk-forward evaluation: realized returns `r_t = w_tᵀ r_{t+1}`, Sharpe ratio, mean, maximum drawdown |
| `src/baseline.py` | SD-relaxation (`SDR_revised`, `make_SDR_solver`), mSSRM-PGA (`mSSRM_PGA`), hard OSCAR |
| `src/model.py` | MLP predictor |
| `src/artifacts.py` | output folder naming and saving |
| `src/aggregate.py` | mean ± std over seeds |
| `src/data_generator.py` | data download |

Modes:

- **dfl** — for each `k`: build the MLP, train with the `dfl` loss (the OSCAR layer is inside
  the loss), evaluate with `oscar`.
- **pfl** — train one MSE predictor per `(dataset, seed)` (reused for every `k` and
  optimizer), then evaluate it with each selected optimizer.
- **historic** — use the mean of each look-back window as the expected return and evaluate
  with each optimizer.

In every mode the covariance of each sample is estimated from the same look-back window
that forms the model input and is held fixed inside the decision layer.

## License

MIT — see [LICENSE](LICENSE).
