# Machine Learning Model Comparison

A small Streamlit app for training and comparing machine learning models on built-in scikit-learn datasets.

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

From the project directory, start the app:

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually <http://localhost:8501>.

## Use

1. Choose a dataset from the dropdown:
   - Iris (classification)
   - Breast Cancer (classification)
   - Diabetes (regression)
2. Select one or more compatible models.
3. Click **Train and compare** to view the performance table and chart.

Classification datasets offer Logistic Regression, Decision Tree, and Random Forest. Their results include Accuracy and weighted F1. The regression dataset offers Linear Regression, Decision Tree, and Random Forest, with R², RMSE, and MAE.

The app uses a 75/25 train/test split with a fixed random state so model comparisons use the same test data.
