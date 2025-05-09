import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pages.Model_Performance_Evaluation import show_evaluation

# Load and prepare data
df = pd.read_csv("telecom_data.csv")  # Make sure this path is correct
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Show evaluation when button is clicked
if st.button("Evaluate Model"):
    show_evaluation(y_test, y_pred, "Random Forest")

# --- UI section below this ---
st.set_page_config(
    page_title="Telco Customer Churn Analysis",
    layout="wide",
    page_icon="🔍"
)

st.title("🌟Welcome to Telco Customer Churn Analysis🌟")
st.write("Use the sidebar to navigate between pages.")

# ... (your intro and description content continues here)
