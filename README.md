# CO₂ Concentration Forecasting with a Custom Transformer

**Applied Computing AI Challenge | Time-series forecasting, model evaluation, and containerized inference**

## Overview

This project implements an end-to-end forecasting and inference workflow for CO₂ concentration at six sampling locations (L1–L6) in an absorber at Imperial College London's carbon-capture pilot plant. The objective is to forecast future concentrations from historical process measurements and available historical CO₂ observations, rather than estimate concentration using measurements from the forecasted future timestamp.

The submission comprises a **PyTorch Transformer implemented from fundamental attention and encoder components**, experiment-level cross-validation, history/horizon benchmarking, scientific evaluation notebooks, and a PostgreSQL-backed FastAPI service deployed with Docker Compose.

**Key result:** On the reserved experiment `140207_1.xlsx`, the selected Transformer achieved **MAE 0.1167 and RMSE 0.2241**, compared with **MAE 0.0764 and RMSE 0.1587** for persistence on the **same 83 paired observed L2–L6 targets**. The Transformer does not outperform persistence on this test. This limitation is reported explicitly and examined in the accompanying analysis.

> **Scope:** This is a reproducible research submission and containerized inference prototype, not a production-hardened or demonstrated real-time plant deployment. The selected combined-input model relies on retrospectively assigned historical CO₂ sampling-location labels; see [Data availability and limitations](#data-availability-and-limitations).

## Challenge coverage

| Challenge area | Implementation |
| --- | --- |
| Custom Transformer | Self-implemented multi-head self-attention, positional encoding, causal masking, encoder blocks, and six-output prediction head in `src/co2_submission/` |
| Forecasting and testing | Experiment-separated training, nested leave-one-experiment-out (LOEO) development evaluation, and a reserved final test |
| Time-window benchmarking | Histories of 12 and 18 steps, horizons of 1 and 3 steps, and three Huber training objectives |
| Reporting and interpretation | Data exploration, model evaluation, and root-cause analysis notebooks |
| Serving | PostgreSQL storage, FastAPI prediction endpoints, and Docker Compose |

## Repository structure

```text
.
├── api/                         # FastAPI application and request validation
├── data/raw/                    # Eight original Excel experiments
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_evaluation.ipynb
│   └── 03_root_cause_analysis.ipynb
├── results/
│   ├── earlystop_cv/            # Archived development-CV results
│   └── final_test/              # Independent test outputs and model checkpoint
├── scripts/                     # Training CLI, ingestion, and inference utilities
├── src/co2_submission/          # Transformer, features, reconstruction, training, deployment
├── tests/                       # Regression and inference-consistency tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt             # Training, analysis, notebook dependencies
├── requirements-deploy.txt      # Container inference dependencies
└── EARLY_STOPPING_GUIDE.md      # Detailed selection and early-stopping protocol
```

The displayed structure describes the intended **clean submission layout**. Keep only one authoritative `03_root_cause_analysis.ipynb` under `notebooks/` when finalizing the repository. The `outputs/` directory and duplicate ZIP archives are not required for reviewing the saved results unless they contain otherwise unavailable artifacts.

## Data and forecasting task

The dataset contains eight separate operating experiments:

`140120_1.xlsx`, `140206_1.xlsx`, `140207_1.xlsx`, `140207_2.xlsx`, `140214_1.xlsx`, `140214_2.xlsx`, `140227_1.xlsx`, and `140313_1.xlsx`.

Process measurements are recorded approximately every 43 seconds. CO₂ observations are **sparse and location-specific**: typically only one absorber sampling location is measured at a given timestamp. Thus, a complete six-location observed profile is generally unavailable at a single instant.

For an origin at time step *t*, the model uses an input history ending at *t* and predicts the six-location concentration vector at *t + h*. No process sensor values from *t + h* are supplied as model inputs. The output is nonnegative by construction.

The original L1 measurements are treated as unreliable. A surrogate L1 target is constructed from the relevant L2 sampling dwell following the original reconstruction convention and is distinguished from a genuine observed L1 value. Other missing target locations may be reconstructed for auxiliary training supervision; reconstructed targets receive reduced weight and are **not** treated as independent ground truth for observed-only evaluation. The surrogate/reconstruction procedure is retrospective and is not an online source of measured future concentrations.

## Model and preprocessing

The architecture is a **direct multi-output Transformer** written in PyTorch without using a prebuilt `nn.Transformer` or `nn.MultiheadAttention` module. Its components include custom scaled dot-product attention, causal self-attention masking, positional encodings, residual encoder blocks, normalization, and a prediction head producing L1–L6 concentration estimates. A nonnegative output transformation prevents physically impossible negative concentrations.

Inputs combine a fixed set of 30 process measurements with historical CO₂ context, including availability/mask and age information. Historical CO₂ features are derived from the recorded sampling sequence; reconstructed future labels are not fed into the input features. Feature selection and normalization are fitted on the relevant training experiments, not on held-out validation or test experiments.

The training pipeline supports three objectives: **raw Huber**, **variance-scaled Huber**, and **noise-aware Huber**. Observed targets are weighted more strongly than reconstructed targets; the selected configuration uses a reconstructed-target weight of **0.05**. The architecture is data-driven: the final release does **not** include a mechanistic simulator or physics-informed loss.

## Model selection and evaluation protocol

`140207_1.xlsx` is reserved as the independent final test. The remaining seven experiments are used for development. Model configurations are evaluated with experiment-level LOEO validation rather than random row-wise splitting, which would mix nearby time-series observations across train and validation sets.

The benchmark grid includes:

| Dimension | Values |
| --- | --- |
| History length | 12, 18 time steps |
| Forecast horizon | 1, 3 time steps |
| Training loss | Raw, variance-scaled, noise-aware Huber |
| Input mode | Combined process and historical CO₂ |
| Model selection | Observed-only validation MAE |
| Early stopping | Maximum 150 epochs; patience 20; minimum improvement 0.0001 |

Within each outer development fold, inner validation determines a suitable training duration. The outer model is then retrained on its available development training experiments for a fixed epoch count determined without inspecting that outer held-out experiment. The final model is trained on all seven development experiments using the development-selected configuration and fixed training duration; the reserved test labels are not used to choose the configuration or stopping epoch.

**Interpretation:** Development CV scores are used for model selection and therefore are not an independent unbiased estimate of the selected model's performance. The reserved test is reported separately. Comparisons between horizons describe different prediction tasks and should not be interpreted as if their difficulty were identical.

### Selected final configuration

| Parameter | Selected value |
| --- | --- |
| History | 12 steps (~8.6 minutes) |
| Horizon | 1 step (~43 seconds) |
| Training objective | Noise-aware Huber |
| Input features | Process sensors + historical CO₂ context |
| Reconstructed target weight | 0.05 |
| Final training epochs | 43 |
| Selection metric | Observed-only MAE |

The exact saved model state and metadata are in `results/final_test/final_model.pt` and `results/final_test/final_training/statistics.json`.

## Independent test results

Performance is assessed against genuine observed L2–L6 concentrations. L1 surrogate values are not counted as genuine observed L1 ground truth.

| Evaluation subset | Model | MAE | RMSE |
| --- | --- | ---: | ---: |
| All 88 available observed Transformer targets | Transformer | 0.1446 | 0.2775 |
| 83 paired observed targets | Transformer | 0.1167 | 0.2241 |
| Same 83 paired observed targets | Persistence | **0.0764** | **0.1587** |

**Paired comparison is the appropriate basis for model-versus-baseline conclusions.** Persistence is a demanding baseline for this short-horizon, slowly evolving series. The Transformer underperforms persistence overall on the independent test. Error is not spatially uniform: L6 is a major contributor to the Transformer test error, with test RMSE approximately **0.5594** and positive mean signed error approximately **+0.4141** on its available observed targets. These location-level values should not be compared with baseline figures computed over different availability masks without pairing observations first.

Results are saved in:

- `results/final_test/final_predictions.csv` — row-level forecasts and baseline predictions.
- `results/final_test/final_metrics.csv` — location-level test metrics.
- `results/final_test/final_model.pt` — trained model and preprocessing metadata.
- `results/earlystop_cv/cv_summary.csv` — development configuration comparison.
- `results/earlystop_cv/outer_predictions.csv` — development fold predictions.
- `results/earlystop_cv/inner_epoch_metrics.csv` — inner-validation learning curves.
- `results/earlystop_cv/selected_config.json` — selected development configuration.

The `results/earlystop_cv/` files are **archived outputs of an earlier completed run**, copied without retraining or recomputation during repository packaging. They should not be represented as newly reproduced by the packaging process.

## Scientific analysis notebooks

1. **`01_data_exploration.ipynb`** — experiment inventory, sensor behavior, sampling schedule, sparse CO₂ observations, physical context, and between-experiment operating-state differences.
2. **`02_model_evaluation.ipynb`** — history/horizon/loss comparison, early stopping, development CV, persistence baselines, location-specific errors, and reserved final-test results.
3. **`03_root_cause_analysis.ipynb`** — diagnostic investigation of forecasting errors, including temporal behavior, sampling history, operating-state novelty, and spatial/location-specific disagreement. Interpret associations as diagnostic evidence; do not claim an identified physical cause without sufficient support.

The notebooks are intended to be read in this order. Saved numerical results allow reviewers to inspect the evaluation without repeating the full nested-CV training grid.

## Quick start: inspect results and notebooks

**Requirements:** Python 3.11 or 3.12 is recommended; the full training environment requires PyTorch and the scientific Python dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name co2-transformer --display-name "CO2 Transformer"
python -m jupyter lab
```

Open the notebooks in `notebooks/` and select the `CO2 Transformer` kernel. Run notebooks from the repository root so relative paths resolve consistently. If the notebook code depends on saved results, retain the documented `results/` directory structure.

## Reproduce model selection (computationally intensive)

Place the eight original `.xlsx` files in `data/raw/`, then run:

```bash
python scripts/run.py \
  --mode cv --raw-dir data/raw --out outputs/reproduced_cv \
  --histories 12 18 --horizons 1 3 \
  --losses raw variance noise --input-modes combined --weights 0.05 \
  --l1-proxy-weight 0.05 --max-epochs 150 --patience 20 \
  --min-delta 0.0001 --selection-metric mae \
  --device cpu --allow-retrospective-labels
```

This is a substantial nested experiment-level training job. The supplied archived CV results can be inspected without rerunning it. The `--allow-retrospective-labels` flag explicitly acknowledges the offline availability assumption discussed below.

To run an **independent new final-test reproduction** after development selection, use a new empty output directory:

```bash
python scripts/run.py \
  --mode final-test --raw-dir data/raw --out outputs/reproduced_final_test \
  --config outputs/reproduced_cv/selected_config.json \
  --device cpu --allow-retrospective-labels
```

The repository already contains the saved final-test artifacts under `results/final_test/`; rerunning final testing is not required to inspect them. Do not overwrite the archived final-test directory. See `EARLY_STOPPING_GUIDE.md` for the detailed selection protocol.

## Containerized PostgreSQL + FastAPI inference

The deployment uses three Compose services: PostgreSQL (`db`), database initialization (`init`), and FastAPI (`api`). The initialization service ingests the original experiments into the database, and the API loads the saved model checkpoint. Compose waits for the database health check and initialization before starting the API.

**Prerequisites:** Docker Desktop / Docker Compose; original data under `data/raw/`; `results/final_test/final_model.pt` present. Run from the repository root:

```bash
docker compose up --build
```

The API is exposed locally at port **8000** and PostgreSQL at port **5432** (bound to `127.0.0.1` by the provided Compose configuration). The default database password is intended for **local demonstration only**; override `POSTGRES_PASSWORD` before using the stack in any shared environment.

Available endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | Check model and database availability |
| GET | `/experiments` | List ingested experiments and row counts |
| POST | `/predict` | Forecast at one origin, selected by row index or exact timestamp |
| POST | `/predict/window` | Forecast at eligible origins in a timestamp interval |

Example requests:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/experiments

curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"experiment":"140207_1.xlsx","origin_row":25}'
```

For timestamp-based inference, submit exactly one of `origin_row` or `timestamp` in `/predict`. Requests with insufficient history are rejected. The service is designed for historical experiment replay and demonstration of database-backed inference; it does not establish that retrospective analyzer labels would be known in a live stream.

To stop the stack:

```bash
docker compose down
```

Use `docker compose down -v` **only** if you intentionally want to delete the PostgreSQL volume and reinitialize the database.

## Tests

With the development environment installed, run:

```bash
PYTHONPATH=src python -m pytest -q tests
```

Some inference-consistency tests depend on the saved final-test files; ensure `results/final_test/` is present. The complete-project test suite and fresh Docker startup should be checked in the final assembled repository before submission. An earlier source-only archive did not contain the required final-test artifacts, so a failure caused solely by those missing files should not be interpreted as a model defect.

## Data availability and limitations

**Retrospective location labels.** The selected combined-input model uses historical CO₂ measurements with sampling-location identity derived from retrospective experiment labels. Although the model's input timestamps precede the prediction target, the real-time availability of those location labels is **not established**. The code requires explicit acknowledgment through `--allow-retrospective-labels`. Consequently, this model is validated as an **offline historical-data forecasting experiment**; unrestricted live deployment would require a verified causal sampling-location signal or an alternative input design.

**Sparse and reconstructed supervision.** Only one location is generally measured at each timestamp. Reconstructed concentrations are lower-confidence auxiliary targets, not simultaneous independent measurements. Observed-only L2–L6 metrics are emphasized; L1 is handled through a surrogate convention.

**Generalization and operating regimes.** Experiments exhibit distinct operating states. LOEO validation assesses transfer between recorded experiments, but does not guarantee performance in new plant conditions. Process-to-CO₂ relationships and spatial error patterns should be interpreted in light of this limited experimental coverage.

**Baseline competitiveness.** The selected Transformer does not beat persistence on the paired reserved test. The project reports this outcome rather than claiming superior predictive accuracy. Potential explanations examined in the notebooks include strong short-horizon persistence, sparse observations, and distribution shift; these are not automatically proven physical root causes.

**Deployment scope.** Docker Compose demonstrates reproducible database-backed inference, not authentication, model monitoring, plant connectivity, or production service-level guarantees.

## Reproducibility and provenance

The source code is separated into model, preprocessing, sequence construction, reconstruction, losses, training, and deployment modules. Random seed and environment information are recorded with evaluation outputs. Training-only feature fitting and experiment-level holdouts are used to limit leakage. The final checkpoint includes model and preprocessing metadata needed by the inference service.

For a minimal, unambiguous submission, retain the source, tests, three authoritative notebooks, `results/earlystop_cv/`, `results/final_test/`, required raw data (subject to redistribution permissions), and deployment files. Avoid duplicate notebook versions, stale `outputs/`, and redundant `results.zip` unless they are explicitly needed and documented.

## Acknowledgments

The challenge and CO₂ absorber data are provided through the Applied Computing AI Challenge and originate from Imperial College London's carbon-capture pilot-plant experiments. The original CO₂ estimation paper motivates the absorber sampling context and L1 reliability treatment; this implementation is a separate, data-driven **forecasting** study rather than a reproduction of the original mechanistic estimation model.
