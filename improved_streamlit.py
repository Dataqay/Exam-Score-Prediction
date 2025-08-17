import streamlit as st
import pandas as pd
import numpy as np # pyright: ignore[reportUnusedImport]
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
import seaborn as sns # pyright: ignore[reportMissingModuleSource]
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # pyright: ignore[reportUnknownVariableType]
from io import BytesIO

# -----------------------------
# App Layout
# -----------------------------
st.set_page_config(page_title="Enhanced Exam Analytics", layout="wide")

st.title("📊 Enhanced Exam Analytics Dashboard")
st.markdown("Explore study habits, analyze data, and predict student exam scores.")

# -----------------------------
# Dataset Loader
# -----------------------------
uploaded_file = st.sidebar.file_uploader("📂 Upload your dataset (Excel or CSV)", type=["xlsx", "csv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file) # pyright: ignore[reportUnknownMemberType]
else:
    st.sidebar.info("Using default dataset `student_scores_2000.xlsx`")
    data = pd.read_excel("student_scores_2000.xlsx") # pyright: ignore[reportUnknownMemberType]

# Train Linear Regression Model
X = data[['StudyHours', 'SleepHours']]
y = data['Score']

model = LinearRegression()
model.fit(X, y) # pyright: ignore[reportUnknownMemberType]
y_pred = model.predict(X)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📈 Analytics", "🎯 Predictions"])

# -----------------------------
# 🏠 Home Tab
# -----------------------------
with tab1:
    st.subheader("📌 Dataset Preview")
    st.dataframe(data.head())

    st.subheader("🎯 Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2_score(y, y_pred):.2f}")
    col2.metric("MAE", f"{mean_absolute_error(y, y_pred):.2f}")
    col3.metric("MSE", f"{mean_squared_error(y, y_pred):.2f}")

# -----------------------------
# 📈 Analytics Tab
# -----------------------------
with tab2:
    st.subheader("📊 Data Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Study Hours vs Score")
        fig1, ax1 = plt.subplots()
        sns.scatterplot(x="StudyHours", y="Score", data=data, ax=ax1)
        sns.regplot(x="StudyHours", y="Score", data=data, ax=ax1, scatter=False, color="red")
        st.pyplot(fig1)

    with col2:
        st.write("Sleep Hours vs Score")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x="SleepHours", y="Score", data=data, ax=ax2) # pyright: ignore[reportUnknownMemberType]
        sns.regplot(x="SleepHours", y="Score", data=data, ax=ax2, scatter=False, color="red") # pyright: ignore[reportUnknownMemberType]
        st.pyplot(fig2)

# -----------------------------
# 🎯 Predictions Tab
# -----------------------------
with tab3:
    st.subheader("🧮 Predict Your Exam Score")

    study_hours = st.slider("Study Hours per day", 0, 12, 5)
    sleep_hours = st.slider("Sleep Hours per day", 0, 10, 6)

    input_data = pd.DataFrame([[study_hours, sleep_hours]], columns=["StudyHours", "SleepHours"])
    predicted_score = model.predict(input_data)[0] # pyright: ignore[reportUnknownMemberType]

    st.success(f"✅ Predicted Score: **{predicted_score:.2f}%**")
    st.write("💡 Tip: Aim for **5–7 study hours** and **6–7 sleep hours** for best results.")

    # Save prediction
    prediction_df = input_data.copy()
    prediction_df["PredictedScore"] = [predicted_score]

    # Download as CSV
    csv = prediction_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Prediction (CSV)",
        data=csv,
        file_name="predicted_score.csv",
        mime="text/csv"
    )

    # Download as Excel
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        prediction_df.to_excel(writer, index=False, sheet_name="Prediction") # pyright: ignore[reportUnknownMemberType]
    st.download_button(
        label="📥 Download Prediction (Excel)",
        data=excel_buffer,
        file_name="predicted_score.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
