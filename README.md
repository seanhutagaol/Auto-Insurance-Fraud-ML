# Auto Insurance Fraud Detection: Comparative Predictive Modeling

This repository features a comprehensive machine learning pipeline engineered to solve complex, highly imbalanced classification problems. The project systematically benchmarks multiple predictive algorithms to identify rare stochastic events (fraud), utilizing rigorous data engineering, hyperparameter tuning, and white-box model interpretability.

## 📊 Methodology & Features
* **Data Engineering & Imputation:** Handles missing data, cleans text irregularities (`?`), and computes custom temporal features (e.g., duration between policy bind date and incident date). Outliers are systematically managed using Z-score filtering.
* **Pipeline Leakage Prevention:** Integrates scikit-learn `Pipeline` and `ColumnTransformer` with `imblearn`'s `SMOTE` to handle severe class imbalance strictly within cross-validation folds, preventing data leakage.
* **Model Benchmarking & Tuning:** Optimizes and compares four distinct classifiers via `GridSearchCV` (optimized for ROC AUC):
  1. **Random Forest** (`RandomForestClassifier`)
  2. **Logistic Regression** (`LogisticRegression` with L1/L2 penalties)
  3. **Support Vector Machines** (`SVC` with linear, RBF, and polynomial kernels)
  4. **XGBoost** (`XGBClassifier`)
* **Model Interpretability:** Employs advanced post-hoc interpretability tools including **Permutation Importance**, **Partial Dependence Plots (PDP)**, and **Individual Conditional Expectation (ICE)** curves to map nonlinear feature relationships.

## 📂 Repository Structure
* `src/data_processing.py` — Data wrangling, missing value treatment, and outlier removal functions.
* `src/pipelines.py` — Feature scaling, encoding, and SMOTE pipeline constructors.
* `src/evaluation.py` — Metric tracking, confusion matrices, and cross-validation execution loops.
* `main_analysis.ipynb` — The primary execution notebook generating model comparisons and visualizations.
