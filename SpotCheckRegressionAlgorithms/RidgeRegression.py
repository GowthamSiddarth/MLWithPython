# Ridge Regression
from pandas import read_csv
from numpy.random import randint
from time import time
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge

filename = "../data/housing.csv"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
         'B', 'LSTAT', 'MEDV']
data_frame = read_csv(filename, delim_whitespace=True, names=names)
values = data_frame.values
x, y = values[:, :13], values[:, 13]

num_of_folds, seed, scoring = 10, randint(int(time())), 'neg_mean_squared_error'
k_fold = KFold(n_splits=num_of_folds, random_state=seed)
model = Ridge()

results = cross_val_score(model, x, y, scoring=scoring, cv=k_fold)
print("Results: " + str(results.mean()))
