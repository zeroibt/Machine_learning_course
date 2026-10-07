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

df = pd.read_csv("student-mat.csv")
print(df.head())
print(df.info())
print(df.shape)


