# Bike Sharing Demand Prediction: A Comparative Regression Study of Environmental vs. Temporal Features

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-yellowgreen.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Accelerated-red.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Project Links & Resources

### Interactive Notebooks (Google Colab)

**- Experiment 1 (Without Time Features):** 

**Colab URL:** https://colab.research.google.com/drive/1UHSLPbnOHMNUB3gnyCHp-f0BcrGyM0G4 

**- Experiment 2 (With Time Features Included):**

**Colab URL:** https://colab.research.google.com/drive/1SenDbRpua2glpqv-FpaG851gmQbxyEKD

### Documentation & Data

**- Project Presentation Link:** 
https://docs.google.com/presentation/d/10Kw8RFW7m2bmFHBiwKsigRC3nsbb0neh/edit?usp=sharing&ouid=116368063423906871146&rtpof=true&sd=true

**- Dataset Link:**
https://drive.google.com/file/d/19rhr1WkBefRU_v5qiF8ihiMczy5Oifon/view?usp=sharing

**- Dataset Details Link:**
https://drive.google.com/file/d/1J2CIoda3w8jMzBgFNCF1mKolh39jDSKn/view?usp=sharing

---

## Project Overview

I have completed this Bike Sharing Demand Prediction regression project using Python in collaboration with **United Network of Professionals (UNP)** as part of my data science internship.

The primary goal of this project is to model and forecast hourly bike rental demand using environmental, seasonal, and temporal features from Washington D.C.'s Capital Bikeshare system (17,379 hourly observations across 2011 and 2012).

A key research question explored in this work is: **To what degree do environmental and weather signals alone determine ridership demand, and what is the quantitative impact of adding temporal features (such as hour of day, month, and day of week)?**

To answer this, two experiments were designed, trained, and benchmarked across 8 machine learning regression algorithms and deep Artificial Neural Networks (ANN):

* **Experiment 1 (`BikeSharing_G6.ipynb`)** – Models trained **strictly on environmental and weather attributes** (temperature, humidity, windspeed, weather condition, and season), excluding all time-based signals.
* **Experiment 2 (`BikeSharing_Time_G6.ipynb`)** – Models trained on **both environmental and engineered temporal attributes** (hour dummy encoding, month dummy encoding, day of week, working day, and year).

The steps, methodology, and experimental results are detailed below.

---

## Team & Institutional Affiliation

* **Academic Institution:** Bhavan's Vivekananda College
* **Program:** Data Science Internship Program
* **Collaborating Organization:** United Network of Professionals (UNP)
* **Project Group:** Group 6
* **Team Members:**
  * **Akhiranand Boddani**
  * **Uppaluru S V N Santhosha Roopa**
  * **Deepthi Hansika**
  * **Akshay Kumar**

---

## Dataset Description

The dataset corresponds to two-year historical logs (2011–2012) from the **Capital Bikeshare System** in Washington, D.C., aggregated hourly and enriched with Freemeteo meteorological data.

* **Total Records:** 17,379 hourly observations
* **Target Variable:** `cnt` (Total hourly bike rentals: `cnt = casual + registered`)
* **Predictor Groups:**
  * **Temporal Features:** `season` (1: Spring, 2: Summer, 3: Fall, 4: Winter), `yr` (0: 2011, 1: 2012), `mnth` (1–12), `hr` (0–23), `holiday` (binary), `weekday` (0–6), `workingday` (binary).
  * **Weather/Environmental Features:** `weathersit` (1: Clear/Few clouds, 2: Mist/Cloudy, 3: Light Rain/Snow, 4: Heavy Rain/Ice Pellets), `temp` (Normalized temperature in Celsius, divided by 41), `atemp` (Normalized feeling temperature in Celsius, divided by 50), `hum` (Normalized humidity, divided by 100), `windspeed` (Normalized wind speed, divided by 67).

### Feature Selection & Data Leakage Prevention

* **Target Leakage Elimination:** The columns `casual` and `registered` were dropped prior to modeling because their sum directly equals `cnt`. Keeping them would represent 100% data leakage.
* **Identifier & Date Pruning:** `instant` (row index) and `dteday` (date string) were removed as they carry no generalizable predictive relationship.
* **Multicollinearity Elimination:** `atemp` (feeling temperature) exhibits near-perfect collinearity with `temp` (`r ≈ 0.99`). `atemp` was dropped to avoid coefficient instability.
* **Redundancy Pruning:** `holiday` was dropped in temporal modeling as it is redundant when `workingday` and `weekday` are present.

---

## Project Methodology & Step-by-Step Execution

### 1. Data Preparation & Cleaning

* **Data Inspection & Missing Value Audit:** Verified dataset dimensions and confirmed that **0 missing/null values** exist across all 17 columns. Data types were converted into suitable formats for numerical scaling and categorical dummy encoding.
* **Target Variable Transformation:** Because `cnt` showed severe positive right-skewness (`skewness > 1.2`), a logarithmic transformation (`log_cnt = ln(cnt + 1) = np.log1p(cnt)`) was applied to stabilize variance, approximate a normal distribution, and prevent negative count predictions when back-transformed via `exp(y_pred) - 1`.
* **Outlier Detection & Removal (IQR):** Applied Interquartile Range filtering (`Q1 - 1.5 × IQR` to `Q3 + 1.5 × IQR`) across numerical features (`temp`: 0, `hum`: 22, `windspeed`: 342, `log_cnt`: 158). Removed **512 outliers** (2.9% of the data), refining the dataset from **17,379 to 16,867 clean records**.

### 2. Feature Analysis & Engineering

* **Correlation & Multicollinearity (VIF):** Evaluated Pearson and Phi-K non-linear correlations alongside **Variance Inflation Factor (VIF)** scores (`season_SPRING`: 3.75, `temp`: 3.06, `season_WINTER`: 2.63, `season_SUMMER`: 1.87, `hum`: 1.60, `weathersit_Snow/Rain`: 1.25, `weathersit_Mist/Cloudy`: 1.16, `windspeed`: 1.13, `workingday`: 1.01). All values remained well below 5.0, confirming coefficient stability.
* **Experiment 1 Encoding (Without Time):** Dropped all temporal columns (`yr`, `mnth`, `hr`, `holiday`, `weekday`) and dummy-encoded `season` and `weathersit` (`drop_first=True`), yielding **9 predictor features**.
* **Experiment 2 Encoding (With Time):** Retained `yr` (binary) and one-hot dummy-encoded `hr`, `mnth`, `weekday`, `workingday`, `season`, and `weathersit`, yielding **74–75 predictor features** to capture commuter peaks and seasonal trends.

### 3. Validation & Model Benchmarking

* **Multi-Split Validation & Scaling:** Evaluated baseline models across **5 train/test splits** (80/20, 75/25, 70/30, 65/35, and 60/40). Fitted `StandardScaler` strictly on training folds before transforming test folds to prevent data leakage.
* **Classical ML Regressors:** Benchmarked 8 algorithms across both experiments:
  1. Linear Regression (OLS)
  2. K-Nearest Neighbors Regressor (KNN)
  3. Support Vector Regressor (Linear SVR)
  4. Decision Tree Regressor
  5. Random Forest Regressor
  6. AdaBoost Regressor
  7. Gradient Boosting Regressor
  8. Extreme Gradient Boosting Regressor (XGBoost)

### 4. Hyperparameter Tuning & Deep Learning (ANN)

* **GridSearchCV Optimization:** Performed 5-Fold Cross Validation (`cv=5`, optimizing `neg_mean_squared_error`) on the 80/20 split:
  * **KNN:** `n_neighbors` (`[3, 5, 7, 9, 11]`), `weights` (`'uniform'`, `'distance'`)
  * **Linear SVR:** `C` (`[0.01, 0.1, 1, 10]`), `epsilon` (`[0.0, 0.1, 0.2]`)
  * **Decision Tree:** `max_depth` (`[3, 5, 10, None]`), `min_samples_split`, `min_samples_leaf`
  * **Random Forest:** `n_estimators` (`[50, 100, 150]`), `max_depth` (`[5, 10, 20]`)
  * **AdaBoost:** `n_estimators` (`[50, 100]`), `learning_rate` (`[0.01, 0.1, 1.0]`)
  * **Gradient Boosting:** `n_estimators` (`[100, 150]`), `learning_rate` (`[0.05, 0.1]`), `max_depth` (`[3, 5]`)
  * **XGBoost:** `n_estimators` (`[100, 150, 200]`), `max_depth` (`[3, 5, 7]`), `learning_rate` (`[0.05, 0.1, 0.2]`)
* **Artificial Neural Network (ANN) Sweeps:** Built `TensorFlow/Keras` models (MSE loss) testing four architectures—**Shallow** (`Input → 10 → 5 → 1`), **Medium** (`Input → 64 → 32 → 16 → 1`), **Deep Pyramid** (`Input → 32 → 32 → 16 → 16 → 8 → 1`), and **Very Deep** (`Input → 128 → 64 → 32 → 16 → 8 → 1`)—across **Adam** and **SGD** optimizers at 20, 50, 100, and 150 epochs.

---

## Key Insights & Discussion

1. **Impact of Temporal Features:**
   * Weather signals alone provide an upper ceiling of `R² ≈ 0.33 to 0.35`. Temperature and weather explain baseline seasonal shifts, but cannot differentiate between high-commute peak hours (8 AM, 5 PM) and off-peak hours (3 AM).
   * Incorporating dummy-encoded hour of the day (`hr`) and calendar features accounts for diurnal human mobility routines, unlocking `R² > 0.93`.
2. **Decision Tree Overfitting:**
   * Unconstrained Decision Trees achieved a training `R²` of 1.00 but performed poorly on test folds (`R² = -0.06` on no-time data). Tree pruning via `GridSearchCV` successfully recovered generalization.
3. **Gradient Boosting vs. Deep Neural Networks:**
   * On low-dimensional environmental data (9 features), tree-based ensembles (XGBoost and Random Forest) outperformed neural networks.
   * On high-dimensional dummy-encoded temporal data (74 features), the deep multi-layer ANN with SGD optimizer achieved the highest overall score (`R² = 0.9368`), closely followed by XGBoost (`R² = 0.9324`).

---

## Future Scope

* **Live Weather API Integration:** Connecting live forecast endpoints (e.g., OpenWeatherMap, Freemeteo) for automated rolling demand inference.
* **Time-Series Deep Learning:** Formulating the prediction as an explicit autoregressive sequential time series (using LSTM, GRU, or Transformers) rather than static tabular regression.
* **Station-Level Prediction:** Extending prediction from system-wide counts to station-specific pickup and dropoff docks to optimize physical bike redistribution logistics.

---

## Acknowledgments

* **United Network of Professionals (UNP):** For providing internship guidance, project mentoring, and structured technical support.
* **Bhavans Vivekananda College:** For academic encouragement and project enablement.
