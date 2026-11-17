# Car Price Prediction

This project explores used-car listing data and trains regression models to estimate a vehicle's listed selling price in Indian rupees (INR). The analysis is implemented in a Jupyter notebook and includes data cleaning, exploratory visualizations, model comparison, and error analysis.

> **Scope:** The data contains historical listings. Predictions are not current market valuations, and results may not generalize to other time periods or markets.

## Dataset

The dataset is the [Vehicle Dataset from CarDekho on Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho). The included source file is `data/cd_v4.csv`. It contains vehicle details such as make and model, year, kilometers driven, fuel type, transmission, location, engine, power, and listed price.

If you replace or restore the data, put a CSV in `data/`. The notebook expects a price column (for example, `Price` or `selling_price`) and supports the CarDekho v4 column names. It prefers filenames containing `v4` when more than one CSV is present.

## Workflow

Run the notebook from the project root, top to bottom. It:

1. Inspects the source data, standardizes columns, removes duplicate and invalid-price records, and handles missing or invalid values.
2. Engineers vehicle features and explores price distributions, vehicle characteristics, brands, and categorical groups.
3. Splits the data into training and test sets (80/20) and builds preprocessing pipelines with median imputation and scaling for numeric features, and most-frequent imputation and one-hot encoding for categorical features.
4. Uses five-fold cross-validation and grid search to compare Linear Regression, Random Forest, and Gradient Boosting.
5. Evaluates the selected model on the held-out test set and creates prediction, residual, and feature-importance diagnostics.

## Results

The current saved comparison selects **Random Forest** by its lowest five-fold cross-validation MAE. On the held-out test set, its MAE is approximately **₹316,782**, RMSE is approximately **₹1,159,949**, and R² is **0.8074**.

The selection criterion is cross-validation MAE; another model may perform better on a different metric. For example, the saved comparison reports a higher test R² for Gradient Boosting. These results depend on the dataset and the notebook's fixed random seed.

## Run the notebook

From the project root, install the required packages and start Jupyter:

```bash
python -m pip install jupyter pandas numpy matplotlib seaborn scikit-learn joblib
jupyter notebook
```

Open `Car_Price_Prediction_ML_Project.ipynb` and run all cells. The notebook reads CSV files from `data/` and writes generated files to `outputs/`.

## Project files

```text
.
├── Car_Price_Prediction_ML_Project.ipynb
├── data/
│   └── cd_v4.csv
└── outputs/
    ├── Best_Car_Price_Model.joblib
    ├── Car_Price_Cleaned_Dataset.csv
    ├── Car_Price_Model_Comparison.csv
    └── charts/
```

Running the full notebook also creates `outputs/Project_Summary.csv` and diagnostic charts in `outputs/charts/`.

## Limitations

- Prices are historical listing values, not verified transaction prices or live market quotes.
- Listing condition, regional demand, duplicate listings, and market changes can affect performance.
- The notebook uses a fixed reference year of 2020 for the engineered vehicle-age feature; update and validate this assumption before using the model for current estimates.
- Predictive relationships and feature importance do not imply causation. Validate against fresh, representative data before relying on predictions.
