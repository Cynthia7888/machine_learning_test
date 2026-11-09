import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")
st.title("Machine Learning Model Comparison")

DATASETS = {
    "Iris (classification)": (load_iris, "classification"),
    "Breast Cancer (classification)": (load_breast_cancer, "classification"),
    "Diabetes (regression)": (load_diabetes, "regression"),
}


def make_models(task):
    if task == "classification":
        return {
            "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        }
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }


dataset_name = st.selectbox("Choose a dataset", list(DATASETS))
load_dataset, task = DATASETS[dataset_name]
dataset = load_dataset()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = dataset.target

models = make_models(task)
selected_models = st.multiselect(
    "Choose model(s) to train",
    options=list(models),
    default=[list(models)[0]],
)

if st.button("Train and compare", type="primary", disabled=not selected_models):
    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    results = []
    for model_name in selected_models:
        model = models[model_name]
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        if task == "classification":
            results.append(
                {
                    "Model": model_name,
                    "Accuracy": accuracy_score(y_test, predictions),
                    "F1 (weighted)": f1_score(y_test, predictions, average="weighted"),
                }
            )
        else:
            results.append(
                {
                    "Model": model_name,
                    "R²": r2_score(y_test, predictions),
                    "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
                    "MAE": mean_absolute_error(y_test, predictions),
                }
            )

    results_df = pd.DataFrame(results).set_index("Model")
    st.subheader("Performance")
    st.dataframe(results_df.style.format("{:.3f}"), use_container_width=True)
    st.bar_chart(results_df)
