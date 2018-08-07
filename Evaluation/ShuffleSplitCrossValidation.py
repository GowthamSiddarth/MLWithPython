# Evaluate using shuffle split cross validation
from pandas import read_csv
from numpy.random import randint
from time import time
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

filename = "../data/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_frame = read_csv(filename, names=names)
values = data_frame.values
x, y = values[:, :8], values[:, 8]

num_of_splits, seed = 10, randint(int(time()))
model = LogisticRegression()
sscv = ShuffleSplit(n_splits=num_of_splits, random_state=seed)
scores = cross_val_score(model, x, y, cv=sscv)

print("Scores: " + str(scores))
