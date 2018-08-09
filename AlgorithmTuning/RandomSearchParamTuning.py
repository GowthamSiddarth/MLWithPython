# Random Search Parameter Tuning
from pandas import read_csv
from numpy.random import randint
from time import time
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge

filename = "../data/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_frame = read_csv(filename, names=names)
values = data_frame.values
x, y = values[:, :8], values[:, 8]

param_grid, seed = dict(alpha=uniform()), randint(int(time()))
model = Ridge()
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, random_state=seed)
random_search.fit(x, y)
print(random_search.best_score_)
print(random_search.best_estimator_.alpha)
