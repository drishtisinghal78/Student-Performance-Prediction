import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓")

st.title("🎓 Student Performance Prediction")
st.write(
    "This machine learning demo estimates a student's final mathematics "
    "grade using study time, absences, and previous grades."
)

@st.cache_resource
def train_model():
    data = fetch_ucirepo(id=320)
    features = data.data.features.copy()
    targets = data.data.targets

    X = features[["studytime", "absences"]].copy()
    X["G1"] = targets["G1"]
    X["G2"] = targets["G2"]
    y = targets["G3"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions),
    }
    return model, metrics

try:
    model, metrics = train_model()
    st.success("Model loaded and trained successfully.")

    st.subheader("Enter Student Details")

    studytime = st.selectbox(
        "Weekly study time category",
        options=[1, 2, 3, 4],
        index=1,
        help="1 = lowest category; 4 = highest category."
    )
    absences = st.number_input(
        "Number of absences", min_value=0, max_value=100,
        value=3, step=1
    )
    g1 = st.slider("First-period grade (G1)", 0, 20, 12)
    g2 = st.slider("Second-period grade (G2)", 0, 20, 13)

    if st.button("Predict Final Grade"):
        student = pd.DataFrame([{
            "studytime": studytime,
            "absences": absences,
            "G1": g1,
            "G2": g2
        }])

        prediction = float(model.predict(student)[0])
        st.subheader("Prediction Result")
        st.metric("Estimated final grade (G3)", f"{prediction:.2f} / 20")
        st.caption(
            "This is a model estimate, not a guaranteed grade. "
            "The model uses previous grades, so G1 and G2 must be known."
        )

    st.divider()
    st.subheader("Model Evaluation")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics['MAE']:.2f}")
    c2.metric("RMSE", f"{metrics['RMSE']:.2f}")
    c3.metric("R²", f"{metrics['R2']:.3f}")
    st.caption("Evaluation uses an 80/20 train-test split with random_state=42.")

    st.subheader("About the Dataset")
    st.write(
        "Dataset: UCI Student Performance, based on students from "
        "Portuguese secondary schools."
    )
    st.markdown(
        "[View the UCI dataset](https://archive.ics.uci.edu/dataset/320/student+performance)"
    )

except Exception as e:
    st.error(
        "The dataset could not be loaded. Please refresh the app and try again. "
        "The app needs internet access to retrieve the UCI dataset."
    )
