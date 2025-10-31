# Ames Housing Price Prediction

This is a comprehensive machine learning project to predict the final sale price of homes in Ames, Iowa, using the Ames Housing Dataset. The project follows a rigorous, end-to-end data science pipeline, from data cleaning and exploratory data analysis (EDA) to feature engineering and comparative model analysis.

The final, optimized model, a **Ridge Regressor**, achieves a **validation Root Mean Squared Logarithmic Error (RMSLE) of 0.12181**.

## 🚀 Project Goal

The primary objective was to build a high-accuracy, interpretable regression model. The project demonstrates a complete end-to-end data science pipeline, with a heavy emphasis on meticulous data cleaning and feature engineering as the primary drivers of model performance.

The pipeline is built to process both the `train.csv` and `test.csv` files, resulting in a model capable of generating a final `submission.csv` file.

## 🧰 Project Workflow

The project is broken down into four key sections, all contained within the main Jupyter Notebook.

### 1. Exploratory Data Analysis (EDA)
* **Target Variable Transformation:** The target, `SalePrice`, was found to be highly right-skewed (skew: 1.88). A `log1p` transformation was applied, creating the normalized `SalePrice_Log` (skew: 0.12), which was used as the target for all models.
* **Outlier Removal:** Visual analysis of `GrLivArea` and `GarageArea` against `SalePrice_Log` identified and removed a few extreme, anomalous outliers.

### 2. Data Processing Pipeline (Sections 3.15 - 3.20)
This is the core of the project. To ensure identical processing and prevent data leakage, the `df_train` and `df_test` sets were combined into a single `df_all` DataFrame.

* **Missing Data Imputation:**
    * **"Fake" `NaN`s:** 18+ columns with "missing" data were identified as having `NaN` as a *category* (e.g., `NaN` in `PoolQC` means "No Pool"). These were systematically filled with strings like `'None'` or `'NoBasement'`.
    * **"Real" `NaN`s:** `LotFrontage` was intelligently imputed using the **median `LotFrontage` of its corresponding `Neighborhood`**. All other numerical `NaN`s (like `MasVnrArea`) were filled with `0`.
* **Feature Engineering (Corrected):** New, high-value features were created from the raw data *before* any transformations:
    * `TotalSF`: `TotalBsmtSF` + `1stFlrSF` + `2ndFlrSF`
    * `TotalBath`: Combined all full and half baths.
    * `HouseAge`: `YrSold` - `YearBuilt`
    * `IsRemodeled`: Binary flag (`YearRemodAdd` != `YearBuilt`)
* **Skew Transformation:** A `log1p` transformation was applied to all numerical features (both original and engineered) with a skew > 0.75.
* **Encoding:** 17 ordinal features were mapped (e.g., `Ex`=5, `Gd`=4), and all remaining nominal features were one-hot encoded using `pd.get_dummies`.
* **Splitting:** The fully processed `df_all` was split back into the model-ready `df_train` and `df_test`.

### 3. Model Training & Analysis (Section 4)
* **Scaling:** A `StandardScaler` was fit *only* on the training data and then applied to the validation and test sets.
* **Model Comparison:** Four different models were trained and compared. The `Ridge` model was the clear winner.

| Model | Validation RMSLE |
| :--- | :--- |
| **Ridge Regression** | **0.12181** |
| LGBM | 0.13315 |
| XGBoost | 0.13340 |
| RandomForest | 0.13943 |

### 4. Key Findings
* **Feature Importance:** The most predictive features, according to the final `Ridge` model, were **`GrLivArea`**, **`MSZoning_RL`** (Residential Low Density), **`MSZoning_RM`** (Medium Density), and **`SaleType_New`**.
* **Engineering > Complexity:** The most significant finding was that the simple, linear `Ridge` model decisively outperformed complex tree-based models (LGBM, XGBoost). This proves that the intensive **data cleaning and feature engineering** were the most important steps in the project.

## 💻 How to Run

This project was built in a Google Colab environment.

### Dependencies
pandas numpy scikit-learn matplotlib seaborn lightgbm xgboost

### Execution
1.  Place the `train.csv`, `test.csv`, and `data_description.txt` files in a known location (e.g., Google Drive).
2.  Open the Jupyter Notebook.
3.  Update the file paths in Section 1 to point to your data.
4.  Run all cells sequentially. The script will automatically process all data, train the models, and generate a final `submission.csv` file.
