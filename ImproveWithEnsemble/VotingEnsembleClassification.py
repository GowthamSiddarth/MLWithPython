# Voting Ensemble for Classification
from pandas import read_csv
from numpy.random import randint
from time import time
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

filename = "../data/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_frame = read_csv(filename, names=names)
values = data_frame.values
x, y = values[:, :8], values[:, 8]

num_of_folds, seed = 10, randint(int(time()))
k_fold = KFold(n_splits=num_of_folds, random_state=seed)

estimators = []
estimators.append(('LogisticRegression', LogisticRegression()))
estimators.append(('SVC', SVC()))
estimators.append(('DecisionTreeClassifier', DecisionTreeClassifier()))

model = VotingClassifier(estimators)
results = cross_val_score(model, x, y, cv=k_fold)
print(results.mean())
