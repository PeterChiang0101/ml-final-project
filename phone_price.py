import pandas as pd
import numpy as np

df = pd.read_csv('data/mobile_price_train.csv')
df.dtypes

df.isnull().sum()
corr = df.corr()
high_corr = corr.index[abs(corr['price_range'])>0.1]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = df[high_corr]
y = df['price_range']

rf = RandomForestClassifier()