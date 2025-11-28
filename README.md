# Salary Prediction Model using Stack Overflow Survey Data

## Project Overview

This project implements a machine learning pipeline to predict the annual converted compensation (`ConvertedCompYearly`) for tech professionals. The analysis leverages data from the **2025 Stack Overflow Annual Developer Survey**.

The methodology encompasses robust data cleaning, feature engineering, and the training of a **CatBoost Regressor**, chosen for its superior performance with mixed-type data (categorical and numerical features).

## Data Source

The raw data used for this project is sourced from the **2025 Stack Overflow Annual Developer Survey**.

*   **Source File:** `data/raw/survey_results_public.csv`

## Methodology and Pipeline

The project follows a standard machine learning workflow, divided into three main stages: Attributes Engineering, Training, and Evaluation.

### 1. Attributes Engineering

This stage focuses on preparing the raw data for model consumption.

| Step | Description | Key Techniques |
| :--- | :--- | :--- |
| **Attribute Selection** | A subset of 16 relevant columns (e.g., Age, Country, Tech Stack, Education Level, Employment) were selected from the 172 available columns. | Feature Subset Selection |
| **Data Cleaning** | Handled missing values, removed outliers from the target variable (`ConvertedCompYearly`) using **Modified Z-Score**, and removed outliers from `YearsCode` using **IQR**. Low-frequency categories in columns like `Country`, `DevType`, and `Employment` were filtered. | Modified Z-Score, IQR Filtering |
| **Attribute Transformation** | Multi-label columns (e.g., `LanguageHaveWorkedWith`, `DatabaseHaveWorkedWith`) were parsed from semicolon-separated strings and transformed into a wide format using **One-Hot Encoding** (dummy variables) to represent the presence of each technology. | One-Hot Encoding, String Splitting |

*   **Processed Data File:** `data/training/dataset.csv`

### 2. Model Training

A CatBoost Regressor was selected for the final prediction task.

*   **Target Transformation:** The target variable (`ConvertedCompYearly`) was transformed using **$\log(1+x)$** (`np.log1p`) to normalize its highly skewed distribution and improve model stability.
*   **Split:** Data was split into 80% for training and 20% for testing, with `random_state=42`.
*   **Model:** CatBoost Regressor (depth=8, learning\_rate=0.5, iterations=2000). CatBoost was utilized for its ability to handle categorical features efficiently without explicit one-hot encoding within the pipeline.

### 3. Evaluation and Results

The model was evaluated using standard regression metrics on the *untransformed* (real dollar) salary scale.

The evaluation metrics achieved on the test set are as follows:

| Metric                             | Value             | Interpretation                                                                          |
|:-----------------------------------|:------------------|:----------------------------------------------------------------------------------------|
| **MAE** (Mean Absolute Error)      | $26,265.90        | On average, the model's prediction is off by this dollar amount.                        |
| **RMSE** (Root Mean Squared Error) | $38,412.38        | The standard deviation of the residuals (prediction errors).                            |
| **$R^2$ Score**                    | 0.575             | Approximately 57.5% of the variance in the salary is explained by the model's features. |

*   **Trained Model File:** `models/catboost-salary-prediction-1.cbm`

## Setup and Dependencies

To reproduce the project, ensure you have the required Python libraries installed using the `uv` package manager.

### Prerequisites

*   Python (3.10+)
*   `uv` for package management

### Installation

Use the following command to install the required libraries:

```bash
# Install the necessary dependencies
uv sync