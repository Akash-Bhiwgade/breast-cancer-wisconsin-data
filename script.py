import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing, model_selection, tree

df = pd.read_csv('data.csv')
df.drop(['id', 'Unnamed: 32'], 1, inplace=True)

df = df.fillna(df.mean())

X = np.array(df.drop(['diagnosis'], axis=1))
y = np.array(df['diagnosis'])
le = preprocessing.LabelEncoder()
le.fit(y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#clf = svm.SVC(kernel='poly')
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)