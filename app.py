import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
from tensorflow import keras

# Load ML and DL models
ml_model = pickle.load(open("model.pkl", "rb"))
dl_model = keras.models.load_model("diabetes_dl_model.h5")

# Page settings
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")

st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🔮 Prediction", "📊 Model Comparison"])

# Prediction Page
if page == "🔮 Prediction":
    st.title("🩺 Diabetes Prediction App")
    st.write("Enter patient details below to check diabetes likelihood")

    with st.form("input_form"):
        col1, col2 = st.columns(2)

        with col1:
            preg = st.number_input("Pregnancies", 0, 20, 1)
            glucose = st.number_input("Glucose", 0, 200, 120)
            bp = st.number_input("Blood Pressure", 0, 140, 70)
            skin = st.number_input("Skin Thickness", 0, 100, 20)

        with col2:
            insulin = st.number_input("Insulin", 0, 900, 79)
            bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
            age = st.number_input("Age", 0, 120, 33)

        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        ml_pred = ml_model.predict(data)[0]
        ml_proba = ml_model.predict_proba(data)[0]

        dl_pred = dl_model.predict(data)[0][0]
        dl_class = int(dl_pred > 0.5)

        st.subheader("🧠 Results")
        st.write(f"**Machine Learning Prediction:** {'Diabetic' if ml_pred==1 else 'Not Diabetic'} (Confidence: {max(ml_proba)*100:.2f}%)")
        st.write(f"**Deep Learning Prediction:** {'Diabetic' if dl_class==1 else 'Not Diabetic'} (Confidence: {dl_pred*100:.2f}%)")

# Model Comparison Page
elif page == "📊 Model Comparison":
    st.title("📊 ML vs DL Performance Comparison")

    # Load dataset for evaluation
    data = pd.read_csv("diabetes.csv")
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    ml_preds = ml_model.predict(X)
    ml_proba = ml_model.predict_proba(X)[:, 1]

    dl_preds = (dl_model.predict(X) > 0.5).astype(int).ravel()
    dl_proba = dl_model.predict(X).ravel()

    from sklearn.metrics import accuracy_score, roc_auc_score

    ml_acc = accuracy_score(y, ml_preds)
    ml_auc = roc_auc_score(y, ml_proba)

    dl_acc = accuracy_score(y, dl_preds)
    dl_auc = roc_auc_score(y, dl_proba)

    st.write(f"**Machine Learning Accuracy:** {ml_acc:.2f}, AUC: {ml_auc:.2f}")
    st.write(f"**Deep Learning Accuracy:** {dl_acc:.2f}, AUC: {dl_auc:.2f}")

    # ROC Curves
    fpr_ml, tpr_ml, _ = roc_curve(y, ml_proba)
    fpr_dl, tpr_dl, _ = roc_curve(y, dl_proba)

    fig, ax = plt.subplots()
    ax.plot(fpr_ml, tpr_ml, label=f"ML (AUC = {ml_auc:.2f})")
    ax.plot(fpr_dl, tpr_dl, label=f"DL (AUC = {dl_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    # Extra Metrics
    ml_precision = precision_score(y, ml_preds)
    ml_recall = recall_score(y, ml_preds)
    ml_f1 = f1_score(y, ml_preds)

    dl_precision = precision_score(y, dl_preds)
    dl_recall = recall_score(y, dl_preds)
    dl_f1 = f1_score(y, dl_preds)

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "ROC-AUC", "Precision", "Recall", "F1-score"],
        "Machine Learning": [ml_acc, ml_auc, ml_precision, ml_recall, ml_f1],
        "Deep Learning": [dl_acc, dl_auc, dl_precision, dl_recall, dl_f1]
    })

    st.subheader("📊 Metrics Comparison (ML vs DL)")
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_df.set_index("Metric").plot(kind="bar", ax=ax)
    plt.xticks(rotation=0)
    plt.ylim(0, 1)
    st.pyplot(fig)
