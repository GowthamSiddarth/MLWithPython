# Grid Search for Algorithm Tuning
import numpy as np
from pandas import read_csv
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

filename = "../data/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_frame = read_csv(filename, names=names)
values = data_frame.values
x, y = values[:, :8], values[:, 8]

alphas = np.array([1, 0.1, 0.001, 0.0001, 0])
param_grid = dict(alpha=alphas)

model = Ridge()
grid = GridSearchCV(model, param_grid=param_grid)
grid.fit(x, y)
print(grid.best_score_)
print(grid.best_estimator_.alpha)
