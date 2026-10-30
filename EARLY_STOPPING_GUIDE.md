# Epoch-by-epoch learning curves and early stopping

This release replaces discrete `--epochs` candidates with **inner-fold early stopping**.
It does not change the Transformer architecture, inputs, reconstruction, loss functions, baseline methods, or frozen test split.

## Training and model selection

- `--losses raw variance noise` selects the **training** Huber objective(s).
- `--selection-metric mae` (or `rmse`) selects epochs using genuine observed L2–L6 validation measurements, with **equal weight per location that has observations**.
- Every inner model logs **every epoch**, up to `--max-epochs`, in `inner_epoch_metrics.csv`. Its `split` column distinguishes `train` from `inner_validation`.
- A fold's best checkpoint is updated only when validation metric decreases by **more than** `--min-delta`. Training stops after `--patience` consecutive non-improving epochs. `stop_reason=max_epochs` means no patience stop; convergence is **not** implied.
- The inner-fold **best checkpoint** is restored before returning from training. To train an outer model on all six outer-training experiments (without a validation set), we select the **rounded median of the inner folds' best epochs** and retrain from scratch for exactly that many epochs. This avoids using outer-test observations to choose the epoch. The same procedure is repeated on all seven development experiments to determine the final training epoch.
- The full seven-fold CV produces `selected_config.json`; partial `--folds` runs intentionally do not.
- The final test never uses its labels for early stopping. It trains on all seven development experiments for the selected fixed epoch count.

## Files to inspect

- `inner_epoch_metrics.csv`: every inner fold's train loss and observed-only validation MAE/RMSE at **each epoch**, plus `selection_score`, `best_epoch`, `patience_counter`, and `improved` on validation rows.
- `early_stopping_summary.csv`: each inner fold's `best_epoch`, `best_score`, `stopped_epoch`, and `stop_reason`.
- `inner_location_objective_metrics.csv`: per-location diagnostics for every epoch.
- `outer_folds/.../epoch_metrics.csv`: training-only curves for each retrained outer model (no outer-test labels used during training).
- `selected_config.json`: selected `epoch` and early-stopping settings, written only after full CV.

## Installation

```bash
cd ~/Downloads/co2_direct_transformer_release_cli_fixed
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
mkdir -p data/raw
# Place all eight original .xlsx files in data/raw, named 140120_1.xlsx, etc.
PYTHONPATH=src pytest -q tests
```

## Small real-data pilot (one outer fold)

```bash
python scripts/run.py \
  --mode cv --raw-dir data/raw --out outputs/earlystop_pilot \
  --histories 12 18 --horizons 1 3 \
  --losses raw --input-modes combined --weights 0.05 \
  --l1-proxy-weight 0.05 --max-epochs 150 --patience 20 \
  --min-delta 0.0001 --selection-metric mae \
  --folds 140120_1.xlsx --device cpu --allow-retrospective-labels
```

## Full seven-fold development experiment

```bash
python scripts/run.py \
  --mode cv --raw-dir data/raw --out outputs/earlystop_cv \
  --histories 12 18 --horizons 1 3 \
  --losses raw variance noise --input-modes combined --weights 0.05 \
  --l1-proxy-weight 0.05 --max-epochs 150 --patience 20 \
  --min-delta 0.0001 --selection-metric mae \
  --device cpu --allow-retrospective-labels
```

**Computational warning:** Nested LOEO trains many independent models. Begin with the pilot and inspect the runtime and learning curves before full CV. Different horizons should be reported separately; comparing their MAE as if they were the same forecasting task is misleading.

## Frozen final test (only after development selection)

```bash
python scripts/run.py \
  --mode final-test --raw-dir data/raw --out outputs/earlystop_final \
  --config outputs/earlystop_cv/selected_config.json \
  --device cpu --allow-retrospective-labels
```

Do **not** pass `--epochs`: it has been removed. Do **not** run final test using a pilot/partial-fold configuration.

## Sharing outputs

```bash
cd outputs
zip -r earlystop_pilot.zip earlystop_pilot -x '*.pt'
```

Upload the ZIP. It includes epoch curves and early-stopping summaries; `.pt` checkpoints are excluded to reduce size.
