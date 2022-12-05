import pandas as pd
import numpy as np

df = pd.read_csv('data/mobile_price_train.csv')
df.dtypes

df.isnull().sum()
corr = df.corr()
high_corr = corr.index[abs(corr['price_range'])>0.1]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = df[high_corr.drop('price_range')]
y = df['price_range']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

prediction = rf.predict(X_test)

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print('Accuracy: ', accuracy_score(y_test, prediction))
print('Recall: ', recall_score(y_test, prediction, average='macro'))
print('Precision: ', precision_score(y_test, prediction, average='macro'))