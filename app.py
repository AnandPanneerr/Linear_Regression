# import joblib
# import os

# def load_artifacts():
#     base = os.path.dirname(__file__)
#     model = joblib.load(os.path.join(base, "model.pkl"))
#     scaler = joblib.load(os.path.join(base, "scaler.pkl"))
#     return model, scaler

# model, scaler = load_artifacts()

# # Take inputs (MATCH TRAINING FEATURES)
# marks = float(input("Enter Marks (0–100): "))
# attendance = float(input("Enter Attendance (%): "))

# # IMPORTANT: same order as training
# data = scaler.transform([[marks, attendance]])

# prediction = model.predict(data)

# if prediction[0] == 1:
#     print("✅ PASS")
# else:
#     print("❌ FAIL")
import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Student Result Predictor", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")
    st.stop()

st.title("🎓 Student Result Predictor")
st.write("Enter student details below:")

marks = st.number_input(
    "Enter Marks (0–100)", min_value=0.0, max_value=100.0, value=50.0
)

attendance = st.number_input(
    "Enter Attendance (%)", min_value=0.0, max_value=100.0, value=75.0
)

if st.button("Predict Result"):
    try:
        input_data = np.array([[marks, attendance]])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)

        if prediction[0] == 1:
            st.success("✅ PASS")
        else:
            st.error("❌ FAIL")

    except Exception as e:
        st.error(f"Prediction Error: {e}")


