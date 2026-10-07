#lab 7 open ended
#Project name :Student
#Roll number# 24f-bsai-068
from statistics import linear_regression

#IMPORTS:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

#dataset initialization:

df = pd.read_csv("student-mat.csv" ,sep=";")
print(df.head())
print(df.info())
print(df.shape)

#data processing
feature_cols = ["studytime", "absences", "G1", "G2"]
df = df.dropna(subset=feature_cols + ["G3"])
X = df[feature_cols].copy()

y_marks=df["G3"].values

#conditional binary 0 and 1 for logistic regression if marks are greater or equal to 10 pass else fail.

y_class = (df["G3"] >= 10).astype(int).values


#Model training
X_train, X_test, y_marks_train, y_marks_test, y_class_train, y_class_test = train_test_split(
    X, y_marks, y_class, test_size=0.2, random_state=42, stratify=y_class
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

print("Shapes of x_train and x_test after scaling:")
print(X_train_scaled.shape)
print(x_test_scaled.shape)

#models to decide pass / fal

lReg = LinearRegression()
lReg.fit(X_train_scaled, y_marks_train)
y_marks_pred = lReg.predict(x_test_scaled)
#passing condition marks should ne greater or equal to 10
lReg_pass_fail = (y_marks_pred >= 10.0).astype(int)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_class_train)
log_reg_pass_fail = log_reg.predict(x_test_scaled)

pass_probabilities = log_reg.predict_proba(x_test_scaled)[:, 1]

#predictions
output_df = pd.DataFrame({
    "Actual Mark (G3)": y_marks_test,
    "Predicted Mark": np.round(y_marks_pred, 2),
    "Predicted Status": [
        "Pass" if c == 1 else "Fail" for c in log_reg_pass_fail
    ],
})

print("\nFinal Marks Prediction vs Actual (First 15 Test Students):")
print(output_df.head(15).to_string(index=False))

mae = np.mean(np.abs(y_marks_test - y_marks_pred))
print(f"\nAverage Prediction Error: ±{mae:.2f} marks")

#model comparision
print("\nLinear Regression")
print("Accuracy :", accuracy_score(y_class_test, lReg_pass_fail))
print("Precision:", precision_score(y_class_test, lReg_pass_fail))
print("Recall   :", recall_score(y_class_test, lReg_pass_fail))

print("\nLogistic Regression")
print("Accuracy :", accuracy_score(y_class_test, log_reg_pass_fail))
print("Precision:", precision_score(y_class_test, log_reg_pass_fail))
print("Recall   :", recall_score(y_class_test, log_reg_pass_fail))

