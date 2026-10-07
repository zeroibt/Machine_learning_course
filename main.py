#lab 7 open ended
#Project name :Student
#Roll number# 24f-bsai-068

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