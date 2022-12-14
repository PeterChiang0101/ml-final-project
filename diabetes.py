import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/diabetes.csv')
df.isnull().sum()
df.dtypes

from sklearn.preprocessing import LabelEncoder

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
corr = df.corr()
high_corr = corr.index[abs(corr['class'])>0.1]

high_corr.drop('class')

from sklearn.model_selection import train_test_split
X = df[high_corr.drop('class')]
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
prediction = lr.predict(X_test)

rc = RandomForestClassifier()
rc.fit(X_train, y_train)
rc_prediction = rc.predict(X_test)


print('Accuracy: ', accuracy_score(y_test, rc_prediction))
print('Recall: ', recall_score(y_test, rc_prediction))
print('Precision: ', precision_score(y_test, rc_prediction))


print('Accuracy: ', accuracy_score(y_test, prediction))
print('Recall: ', recall_score(y_test, prediction))
print('Precision: ', precision_score(y_test, prediction))

import joblib

joblib.dump(rc, 'models/diabetes_lr.pkl', compress=3)