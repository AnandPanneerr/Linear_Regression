import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\Anand\OneDrive - Rathinam Group Of Institutions\Desktop\Linear Alogrithim\Data_set\Student_Result_5000_Dataset.csv")  
# ⚠️ Make sure this CSV exists in the same folder

# Features & Target
X = df.drop("Result", axis=1)   # example: Marks, Attendance, etc.
y = df["Result"]               # 1 = Pass, 0 = Fail

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_scaled, y)

# Save model & scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ model.pkl and scaler.pkl created successfully")
