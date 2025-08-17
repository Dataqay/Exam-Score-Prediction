import streamlit as st
import pandas as pd
import numpy as np  # pyright: ignore[reportUnusedImport]
from sklearn.linear_model import LinearRegression  # pyright: ignore[reportUnusedImport]

# Load dataset
data = pd.read_excel("C:/Users/DATAQAY/my_project/pridiction_2/student_scores_2000.xlsx") # pyright: ignore[reportUnknownMemberType]



# Calculate statistical properties
study_mean, study_std = data['StudyHours'].mean(), data['StudyHours'].std()
sleep_mean, sleep_std = data['SleepHours'].mean(), data['SleepHours'].std()
score_mean, score_std = data['Score'].mean(), data['Score'].std()

# Generate synthetic data for 200 rows
np.random.seed(42)  # For reproducibility
expanded_data = pd.DataFrame({
    'StudyHours': np.clip(np.random.normal(study_mean, study_std, 200).round().astype(int), 1, 12),
    'SleepHours': np.clip(np.random.normal(sleep_mean, sleep_std, 200).round().astype(int), 4, 8),
    'Score': np.clip(np.random.normal(score_mean, score_std, 200).round().astype(int), 50, 100)
})

# Ensure no negative or unrealistic values
expanded_data['StudyHours'] = expanded_data['StudyHours'].clip(lower=1, upper=12) # pyright: ignore[reportUnknownMemberType]
expanded_data['SleepHours'] = expanded_data['SleepHours'].clip(lower=4, upper=8) # pyright: ignore[reportUnknownMemberType]
expanded_data['Score'] = expanded_data['Score'].clip(lower=50, upper=100) # type: ignore

# Train the model
X = data[['StudyHours', 'SleepHours']]
y = data['Score']
model = LinearRegression()
model.fit(X, y) # pyright: ignore[reportUnknownMemberType]

# Streamlit UI
st.title("📚 Student Exam Score Predictor")
st.write("Adjust the sliders to predict your score based on study & sleep hours.")

# Show dataset in Streamlit
st.write("Preview of Dataset:")
st.dataframe(data.head())  # pyright: ignore[reportUnknownMemberType]



# User input sliders
study_hours = st.slider("Study Hours per day", 0, 10, 5)
sleep_hours = st.slider("Sleep Hours per day", 0, 10, 6)

# Prediction
input_data = pd.DataFrame([[study_hours, sleep_hours]], columns=["StudyHours", "SleepHours"])
predicted_score = model.predict(input_data)[0] # pyright: ignore[reportUnknownMemberType]

# Output
st.subheader(f"Predicted Score: {predicted_score:.2f}%")
st.write("📌 Tip: Aim for **5–7 hours study** and **6–7 hours sleep** for better results.")

# Show dataset
with st.expander("See training dataset"):
    st.dataframe(data) # pyright: ignore[reportUnknownMemberType]
