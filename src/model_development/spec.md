# Polymarket BTC Outcome Prediction: Technical Specification

## 1. Project Overview
The objective is to build a machine learning pipeline that predicts the outcome of 15-minute Bitcoin (BTC) prediction markets on Polymarket. The models will predict the probability of the market resolving as "UP" based on the state of the order book and the underlying asset price. The pipeline must train both an interpretable white-box model and a complex black-box model, evaluate them against the market's own baseline, and apply explainability frameworks.

## 2. Data Processing Requirements

### 2.1 Filtering & Target Definition
- **Input Data:** A Parquet file containing historical order book snapshots for various crypto prediction markets.
- **Market Filter:** The dataset contains multiple assets. Filter the data to include exclusively Bitcoin markets (where the market identifier/slug contains "btc-updown").
- **Target Variable:** The target must be a binary resolution (1 if the market resolved UP, 0 if it resolved DOWN). This should be derived from the final outcome price of the market, discarding any invalid or unresolved states.

### 2.2 Sub-sampling
- The raw data is tracked at 1-second intervals. To eliminate severe autocorrelation and reduce data bloat, downsample the dataset to **1-minute intervals**.
- For each minute in a given 15-minute market, extract the last known state (e.g., the snapshot at second 59, 119, 179, etc.).

### 2.3 Feature Engineering
Do not use raw absolute prices or sizes, as these do not generalize across different price regimes. Engineer the following relative features:
- **Spread:** The difference between the lowest ask price and the highest bid price.
- **Mid-Price (Crowd Probability):** The average of the highest bid price and lowest ask price. This serves as the crowd's current implied probability.
- **Order Book Imbalance (OBI):** The normalized difference between total bid volume and total ask volume across all available levels. Formula: (Total Bid - Total Ask) / (Total Bid + Total Ask).
- **Normalized Price Change:** The percentage change in the underlying asset's price compared to its price at the start of that specific 15-minute market window.
- **Time Remaining:** The number of seconds remaining until the market resolves (assuming a standard 900-second lifespan).

## 3. Modeling Requirements

### 3.1 Data Splitting (Avoid Data Leakage)
- **Strict Chronological Split:** Sort all unique markets by their start time.
- Split the markets into three continuous time blocks:
  - **70% Training Set**
  - **15% Validation Set** (for hyperparameter tuning and early stopping)
  - **15% Test Set** (strictly out-of-sample for final evaluation)
- Never randomly shuffle the overall rows, as this will cause future data to leak into the training set.

### 3.2 Model Architectures
- **White-box Model:** Train a Logistic Regression model to serve as a highly interpretable, linear baseline.
- **Black-box Model:** Train an XGBoost (or LightGBM) classifier to capture non-linear relationships. Utilize the validation set for early stopping to prevent overfitting.
- **Crowd Baseline:** Define a third "model" that simply outputs the `mid_price` feature. This represents the market consensus.

## 4. Evaluation Metrics
Evaluate all three predictors (Logistic Regression, Black-box, and Crowd Baseline) exclusively on the out-of-sample Test Set.
- Calculate and report the Area Under the Receiver Operating Characteristic Curve (AUROC).
- Calculate and report the Area Under the Precision-Recall Curve (AUPR).
- The goal is to observe if the machine learning models can achieve higher AUROC/AUPR than the Crowd Baseline.

## 5. Explainability Requirements

### 5.1 Global Explainability
- Use the SHAP (SHapley Additive exPlanations) framework to calculate global feature importances for the Black-box model across the test set.
- Generate and save a SHAP Summary Plot.

### 5.2 Local Explainability
- Utilize both LIME (Local Interpretable Model-agnostic Explanations) and SHAP to explain individual, specific predictions from the Test Set.
- Select two specific snapshots to explain:
  - **Scenario A (Consensus):** A snapshot where both the Crowd Baseline and the Black-box model confidently and correctly predicted the final outcome.
  - **Scenario B (Contrarian):** A snapshot where the Black-box model's prediction significantly diverged from the Crowd Baseline, and the Black-box model was correct.
- Output the feature weights/impacts from both LIME and SHAP for these two scenarios to compare how they explain the local decision.

## 6. Artifact Generation
The pipeline must output and persist the following artifacts to specific directories:
- **Processed Data:** Save the downsampled, feature-engineered dataset to an `./outputs/` directory.
- **Saved Models:** Serialize and save the trained White-box and Black-box models to a `./models/` directory.
- **Plots:** Save the SHAP Summary Plot image to the `./outputs/` directory.
