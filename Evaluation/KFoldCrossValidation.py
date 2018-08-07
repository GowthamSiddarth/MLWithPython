# Evaluate using K Fold Cross Validation
from pandas import read_csv
from sklearn.model_selection import KFold
from numpy.random import randint
from time import time
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

filename = "../data/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_frame = read_csv(filename, names=names)
values = data_frame.values
x, y = values[:, :8], values[:, 8]

num_of_folds, seed = 3, randint(int(time()))
k_fold = KFold(n_splits=num_of_folds, random_state=seed)
model = LogisticRegression()
scores = cross_val_score(model, x, y, cv=k_fold)
print("Scores: " + str(scores))
