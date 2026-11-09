from io import StringIO

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


def get_metrics(task, actual, predicted):
    if task == "classification":
        return {
            "Accuracy": accuracy_score(actual, predicted),
            "F1 (weighted)": f1_score(actual, predicted, average="weighted"),
        }
    return {
        "R²": r2_score(actual, predicted),
        "RMSE": mean_squared_error(actual, predicted) ** 0.5,
        "MAE": mean_absolute_error(actual, predicted),
    }


def convert_validation_target(target, task, dataset, y):
    if task == "classification":
        label_map = {
            str(label).strip().casefold(): index
            for index, label in enumerate(getattr(dataset, "target_names", []))
        }
        target = target.map(
            lambda value: label_map.get(str(value).strip().casefold(), value)
        )

    converted = pd.to_numeric(target, errors="coerce")
    if converted.isna().any():
        return None, "The target column must contain valid numeric values or known class names."

    if task == "classification" and not converted.isin(set(y)).all():
        return None, "Classification targets must match the selected dataset's class codes or names."

    return converted, None


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

training_key = (dataset_name, tuple(selected_models))
if st.button("Train and compare", type="primary", disabled=not selected_models):
    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    results = []
    fitted_models = {}
    for model_name in selected_models:
        model = models[model_name]
        model.fit(X_train, y_train)
        fitted_models[model_name] = model
        predictions = model.predict(X_test)
        results.append({"Model": model_name, **get_metrics(task, y_test, predictions)})

    st.session_state["training_key"] = training_key
    st.session_state["trained_models"] = fitted_models
    st.session_state["test_results"] = pd.DataFrame(results).set_index("Model")

if st.session_state.get("training_key") == training_key:
    st.subheader("Held-out test performance")
    test_results = st.session_state["test_results"]
    st.dataframe(test_results.style.format("{:.3f}"), use_container_width=True)
    st.bar_chart(test_results)

    validation_tab, input_tab = st.tabs(["Validate labeled CSV", "Predict from text input"])

    with validation_tab:
        uploaded_file = st.file_uploader(
            "Upload unseen labeled data (CSV)",
            type=["csv"],
            help="Include every feature column from the selected dataset and a target column.",
        )
        if uploaded_file is not None:
            try:
                validation_df = pd.read_csv(uploaded_file)
            except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
                st.error(f"Could not read the validation CSV: {error}")
            else:
                if validation_df.empty or validation_df.columns.empty:
                    st.error("The validation CSV must contain at least one labeled data row.")
                else:
                    target_candidates = {"target", "label", "class"}
                    default_target_index = next(
                        (
                            index
                            for index, column in enumerate(validation_df.columns)
                            if str(column).strip().casefold() in target_candidates
                        ),
                        len(validation_df.columns) - 1,
                    )
                    target_column = st.selectbox(
                        "Target column in validation CSV",
                        options=list(validation_df.columns),
                        index=default_target_index,
                    )
                    missing_features = [
                        column for column in X.columns if column not in validation_df.columns
                    ]
                    if missing_features:
                        st.error(
                            f"Validation CSV is missing feature columns: {', '.join(missing_features)}"
                        )
                    elif target_column in X.columns:
                        st.error("Choose a target column that is separate from the feature columns.")
                    else:
                        validation_X = validation_df.loc[:, X.columns].apply(
                            pd.to_numeric, errors="coerce"
                        )
                        if validation_X.isna().any().any():
                            st.error(
                                "Validation feature columns must contain numeric values with no missing cells."
                            )
                        else:
                            validation_y, target_error = convert_validation_target(
                                validation_df[target_column], task, dataset, y
                            )
                            if target_error:
                                st.error(target_error)
                            else:
                                validation_results = []
                                trained_models = st.session_state["trained_models"]
                                for model_name, model in trained_models.items():
                                    predictions = model.predict(validation_X)
                                    validation_results.append(
                                        {
                                            "Model": model_name,
                                            **get_metrics(task, validation_y, predictions),
                                        }
                                    )
                                validation_results_df = pd.DataFrame(
                                    validation_results
                                ).set_index("Model")
                                st.subheader("Unseen data validation")
                                st.dataframe(
                                    validation_results_df.style.format("{:.3f}"),
                                    use_container_width=True,
                                )
                                st.bar_chart(validation_results_df)

    with input_tab:
        feature_names = list(X.columns)
        st.caption("Enter one or more comma-separated rows without a header.")
        with st.expander("Expected feature order"):
            st.code(", ".join(feature_names))
        manual_input = st.text_area(
            "Unseen feature values",
            placeholder=", ".join(["value"] * len(feature_names)),
            key="manual_unseen_input",
        )
        if st.button("Predict input", disabled=not manual_input.strip()):
            try:
                input_data = pd.read_csv(StringIO(manual_input), header=None)
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
                st.error(f"Could not read the input values: {error}")
            else:
                if input_data.shape[1] != len(feature_names):
                    st.error(f"Each row must contain exactly {len(feature_names)} feature values.")
                else:
                    input_data.columns = feature_names
                    input_data = input_data.apply(pd.to_numeric, errors="coerce")
                    if input_data.isna().any().any():
                        st.error("All feature values must be numeric, with no missing values.")
                    else:
                        prediction_rows = []
                        trained_models = st.session_state["trained_models"]
                        for model_name, model in trained_models.items():
                            predictions = model.predict(input_data)
                            for row_number, prediction in enumerate(predictions, start=1):
                                if task == "classification":
                                    target_names = getattr(dataset, "target_names", [])
                                    if 0 <= int(prediction) < len(target_names):
                                        prediction = target_names[int(prediction)]
                                prediction_rows.append(
                                    {
                                        "Row": row_number,
                                        "Model": model_name,
                                        "Prediction": prediction,
                                    }
                                )
                        st.dataframe(pd.DataFrame(prediction_rows), use_container_width=True)
else:
    st.info("Train and compare models to enable unseen-data validation and prediction.")
