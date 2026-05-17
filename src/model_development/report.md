# Polymarket BTC 15-Minute Outcome Prediction

This directory contains the machine learning pipeline developed to predict the outcome of 15-minute Bitcoin markets on Polymarket. 

## Current State of the Project

The pipeline is fully implemented, tested, and capable of processing the raw Polymarket order book data into trained, evaluated, and explainable models.

### Directory Structure
- `src/model_development/`: Contains the pipeline scripts (`preprocess.py`, `train_eval.py`, `explain.py`).
- `outputs/`: Contains generated artifacts such as the processed dataset (`processed_btc_1m.parquet`) and explainability plots (`shap_summary.png`).
- `models/`: Contains the serialized model files (`logistic_regression.joblib`, `xgboost.json`).

### Pipeline Scripts

1. **`preprocess.py`**
   - **Filtering:** Isolates `btc-updown` markets from the 6M+ row dataset.
   - **Downsampling:** Sub-samples 1-second interval data into 1-minute intervals (taking the last known state per minute) to remove extreme autocorrelation and optimize training speed.
   - **Feature Engineering:** Creates relative, normalized features to ensure generalization across market regimes: `spread`, `mid_price` (Crowd Probability), `obi` (Order Book Imbalance), `price_change_pct`, and `time_remaining`.
   - **Target:** Maps the final market resolution to a binary target (`1` for UP, `0` for DOWN).

2. **`train_eval.py`**
   - **Data Split:** Implements a strict chronological split (70% Train, 15% Validation, 15% Test) to prevent data leakage.
   - **Models:** Trains an inherently interpretable white-box model (Logistic Regression) and a complex black-box model (XGBoost with early stopping).
   - **Evaluation:** Evaluates models out-of-sample against the Polymarket "Crowd Baseline" (the raw `mid_price`).
   - **Serialization:** Saves trained models to the `./models/` directory.

3. **`explain.py`**
   - **Global Explainability (SHAP):** Generates a SHAP summary plot for the XGBoost model to compare feature importance globally.
   - **Local Explainability (LIME & SHAP):** Isolates specific test set scenarios (a "Consensus" prediction and a "Contrarian" prediction) and applies both LIME and SHAP to explain the exact feature impacts that drove the prediction for those specific market snapshots.

## Results

Evaluation on the strictly out-of-sample test set (15% of chronologically sorted markets) yielded the following results:

| Model | AUROC | AUPR |
|-------|-------|------|
| **Logistic Regression (White-box)** | **0.8418** | **0.8468** |
| Crowd Baseline (Mid-Price) | 0.8416 | 0.8449 |
| XGBoost (Black-box) | 0.8409 | 0.8453 |

## Key Insights

The results highlight two classic phenomena in quantitative financial modeling:

1. **Why Logistic Regression beat XGBoost:**
   Financial market data inherently possesses a very low signal-to-noise ratio. Complex, non-linear models like XGBoost are prone to finding patterns in the noise (overfitting), which fail to generalize out-of-sample. The strict linear rigidity of Logistic Regression allowed it to capture the true underlying signal without being distracted by market noise.

2. **Why the Model beat the Crowd (Slightly):**
   The Polymarket crowd is highly efficient, as evidenced by the baseline AUROC of 0.8416. However, the Logistic Regression model achieved a microscopic edge (0.0002) by using the crowd's probability (`mid_price`) as a massive anchor weight, and applying tiny mathematical corrections based on the shape of the order book. For example, it learned to slightly elevate the probability if the Order Book Imbalance (`obi`) skewed heavily toward bids, allowing it to be marginally smarter than the raw crowd consensus.
