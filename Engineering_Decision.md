# Engineering Decisions

## Engine-wise splitting
Rows from the same engine are kept in the same split. This prevents the model from seeing part of an engine during training and another part during testing.

## Cycle feature
`cycle` is kept in the final model because it is available at prediction time and the ablation study showed better internal holdout performance when it is included. It is treated as an important age-related feature, not hidden as if it were an independent health measurement.

## Lag and rolling features
One-cycle sensor lags and three-cycle rolling means are used to capture short-term changes in engine behaviour. They are calculated within each engine so information does not cross engine boundaries.

## Scaling
Only Linear Regression uses `StandardScaler`. Tree-based models do not need feature scaling, so no scaler is loaded during XGBoost inference.

## RUL target cap
The final XGBoost model is trained with `RUL` capped at 125 cycles. This keeps the model focused on the range that is most useful for maintenance prediction and reduces large optimistic RUL estimates. The cap is configurable in `configs/config.yaml`.

## Final evaluation
The internal engine-wise holdout is used during development. The NASA test set, with its separate RUL files, is the final unseen benchmark. Because the final model uses a capped target, its internal all-row metrics are not directly comparable with the uncapped baseline results.

The 125-cycle cap was selected after benchmark experimentation on the public CMAPSS test labels. This should be disclosed when presenting the project as a benchmark result.

## Maintenance thresholds
Predicted RUL of 30 cycles or less is classified as Critical. Predictions from 31 to 60 cycles are Warning; values above 60 are Healthy. These are dashboard decision bands, not calibrated probabilities of failure.

## Reproducibility
Random seeds are kept in `configs/config.yaml` and reused by the split and model-training steps.
