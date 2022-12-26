import joblib
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data/winequality-red.csv')
df.isnull().sum()
df.dtypes
df.head()
df['quality'].value_counts()
def draw_hist(df, col, f_num=1):
    acidity = {}
    for i in range(len(df)):
        round_up = round(df[col][i], f_num)
        if round_up in acidity:
            acidity[round_up] += 1
        else:
            acidity[round_up] = 1
    grade3 = {}
    grade4 = {}
    grade5 = {}
    grade6 = {}
    grade7 = {}
    grade8 = {}
    grade = [{}]
    for i in acidity.keys():
        grade3[i] = 0
        grade4[i] = 0
        grade5[i] = 0
        grade6[i] = 0
        grade7[i] = 0
        grade8[i] = 0

    for i in range(len(df)):
        round_up = round(df[col][i], f_num)
        if df['quality'][i] == 3:
            if round_up in grade3:
                grade3[round_up] += 1
            else:
                grade3[round_up] = 1
        elif df['quality'][i] == 4:
            if round_up in grade4:
                grade4[round_up] += 1
            else:
                grade4[round_up] = 1
        elif df['quality'][i] == 5:
            if round_up in grade5:
                grade5[round_up] += 1
            else:
                grade5[round_up] = 1
        elif df['quality'][i] == 6:
            if round_up in grade6:
                grade6[round_up] += 1
            else:
                grade6[round_up] = 1
        elif df['quality'][i] == 7:
            if round_up in grade7:
                grade7[round_up] += 1
            else:
                grade7[round_up] = 1
        elif df['quality'][i] == 8:
            if round_up in grade8:
                grade8[round_up] += 1
            else:
                grade8[round_up] = 1
    for i in range(len(acidity)):
        grade3[list(acidity.keys())[i]] = grade3[list(acidity.keys())[i]] / acidity[list(acidity.keys())[i]] * 100
        grade4[list(acidity.keys())[i]] = grade4[list(acidity.keys())[i]] / acidity[list(acidity.keys())[i]] * 100
        grade5[list(acidity.keys())[i]] = grade5[list(acidity.keys())[i]] / acidity[list(acidity.keys())[i]] * 100
        grade6[list(acidity.keys())[i]] = grade6[list(acidity.keys())[i]] / acidity[list(acidity.keys())[i]] * 100
        grade7[list(acidity.keys())[i]] = grade7[list(acidity.keys())[i]] / acidity[list(acidity.keys())[i]] * 100
        grade8[list(acidity.keys())[i]] = grade8[list(acidity.keys())[i]] / acidity[list(acidity.keys())[i]] * 100

    grade = [grade3, grade4, grade5, grade6, grade7, grade8]
    for i in range(len(grade)):
        grade[i] = dict(sorted(grade[i].items(), key=lambda item: item[0]))

    plt.bar(range(len(grade[0])), list(grade[0].values()),
            align='center', tick_label=list(grade[0].keys()), label='3', color='r')
    plt.bar(range(len(grade[1])), list(grade[1].values()),
            align='center', tick_label=list(grade[1].keys()), label='4', bottom=np.array(list(grade[0].values())), color='orange')
    plt.bar(range(len(grade[2])), list(grade[2].values()),
            align='center', tick_label=list(grade[2].keys()), label='5', bottom=np.array(list(grade[0].values()))+np.array(list(grade[1].values())), color='y')
    plt.bar(range(len(grade[3])), list(grade[3].values()),
            align='center', tick_label=list(grade[3].keys()), label='6', bottom=np.array(list(grade[0].values()))+np.array(list(grade[1].values()))+np.array(list(grade[2].values())), color='g')
    plt.bar(range(len(grade[4])), list(grade[4].values()),
            align='center', tick_label=list(grade[4].keys()), label='7', bottom=np.array(list(grade[0].values()))+np.array(list(grade[1].values()))+np.array(list(grade[2].values()))+np.array(list(grade[3].values())), color='b')
    plt.bar(range(len(grade[5])), list(grade[5].values()),
            align='center', tick_label=list(grade[5].keys()), label='8', bottom=np.array(list(grade[0].values()))+np.array(list(grade[1].values()))+np.array(list(grade[2].values()))+np.array(list(grade[3].values()))+np.array(list(grade[4].values())), color='purple')
    plt.legend([3, 4, 5, 6, 7, 8])
    plt.xlabel(col)
    plt.ylabel('Percentage (%)')

draw_hist(df, 'volatile acidity')
draw_hist(df, 'alcohol', 1)
acidity = dict(sorted(acidity.items(), key=lambda item: item[0]))
plt.bar(range(len(acidity)), list(acidity.values()),
        align='center', tick_label=list(acidity.keys()))
plt.bar(df['volatile acidity'], sum(df['quality'] == 4),
        width=0.1, bottom=df['quality'] == 3, label='4')
plt.hist(df['quality'], bins=5)
plt.title('Quality of Wine')

corr = df.corr()
high_corr = corr.index[abs(corr['quality']) > 0.1]

X = df[high_corr.drop('quality')]
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4)


lr = LogisticRegression(max_iter=100000)
lr.fit(X_train, y_train)
prediction = lr.predict(X_test)

rc = RandomForestClassifier()
rc.fit(X_train, y_train)
rc_prediction = rc.predict(X_test)


print('Accuracy: ', accuracy_score(y_test, rc_prediction))
print('Recall: ', recall_score(
    y_test, rc_prediction, average='micro', zero_division=1))
print('Precision: ', precision_score(
    y_test, rc_prediction, average='macro', zero_division=1))
print(confusion_matrix(y_test, rc_prediction))
sns.heatmap(confusion_matrix(y_test, rc_prediction),
            annot=True, xticklabels=[3, 4, 5, 6, 7, 8], yticklabels=[3, 4, 5, 6, 7, 8])

print('Accuracy: ', accuracy_score(y_test, prediction))
print('Recall: ', recall_score(
    y_test, prediction, average='micro', zero_division=1))
print('Precision: ', precision_score(
    y_test, prediction, average='macro', zero_division=1))
print(confusion_matrix(y_test, prediction))
prob = lr.predict_proba(X_test)
sns.heatmap(confusion_matrix(y_test, prediction),
            annot=True, xticklabels=[3, 4, 5, 6, 7, 8], yticklabels=[3, 4, 5, 6, 7, 8])

joblib.dump(rc, 'models/wine_quality.pkl', compress=3)
