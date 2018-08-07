# Cross Validation Regression MSE
from pandas import read_csv
from numpy.random import randint
from time import time
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

filename = "../data/housing.csv"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
         'B', 'LSTAT', 'MEDV']
data_frame = read_csv(filename, names=names)
values = data_frame.values
x, y = values[:, :8], values[:, 8]

num_of_folds, seed, scoring = 10, randint(int(time())), 'neg_mean_squared_error'
k_fold = KFold(n_splits=num_of_folds, random_state=seed)

model = LinearRegression()
results = cross_val_score(model, x, y, scoring=scoring, cv=k_fold)
print("MSE: %s, %s" % (results.mean(), results.std()))
