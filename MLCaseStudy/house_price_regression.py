# Load libraries
from time import time
from random import randint
from numpy import arange
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.pipeline import Pipeline

# Load dataset
filename = '../data/housing.csv'
attributes_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
                    'B', 'LSTAT', 'MEDV']
data_frame = read_csv(filename, names=attributes_names, delim_whitespace=True)

# Summarize dataset
print("Shape of dataset: %s" % str(data_frame.shape))

print("DataTypes of attributes")
print(data_frame.dtypes)

peek_size = 10
print("Peek size = %d" % peek_size)
print(data_frame.head(n=peek_size))

print("Data Summary")
set_option('display.width', 500)
set_option('precision', 3)
print(data_frame.describe())

print("Skew Summarize Data")
print(data_frame.skew())

print("Correlation of data_frame")
print(data_frame.corr(method='pearson'))

# Data Visualization
'''Unimodal Visualization: Visualizing each attribute separately'''
data_frame.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
data_frame.plot(kind='density', subplots=True, layout=(4, 4), sharey=False, legend=True, fontsize=1)
data_frame.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, sharey=True, fontsize=8)

'''Multimodal Visualization: Visualizing attributes simultaneously'''
scatter_matrix(data_frame)

fig = plt.figure()
axis = fig.add_subplot(111)
caxis = axis.matshow(data_frame.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(caxis)
ticks = arange(0, 14, 1)
axis.set_xticks(ticks)
axis.set_yticks(ticks)
axis.set_xticklabels(attributes_names)
axis.set_yticklabels(attributes_names)

plt.show()

# Split Dataset
values, num_of_features = data_frame.values, len(attributes_names) - 1
X, y = values[:, :num_of_features], values[:, num_of_features]
validation_size, seed = 0.2, randint(0, int(time()))

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=validation_size, random_state=seed)

# Evaluation of Algorithms
num_of_folds, scoring = 10, 'neg_mean_squared_error'
models = [('LR', LinearRegression()), ('Lasso', Lasso()), ('EN', ElasticNet()), ('SVR', SVR()),
          ('CART', DecisionTreeRegressor()), ('kNN', KNeighborsRegressor())]

algo_names, algo_results = [], []
for algo_name, algo in models:
    k_fold = KFold(n_splits=num_of_folds, random_state=seed)
    algo_result = cross_val_score(estimator=algo, X=X_train, y=y_train, cv=k_fold, scoring=scoring)
    algo_results.append(algo_result)
    algo_names.append(algo_name)
    print("Algo: %s Result: %f (mean) %f (std)" % (algo_name, algo_result.mean(), algo_result.std()))

# COmpare Algorithms
fig = plt.figure()
fig.suptitle("Algorithm Comparison")
axis = fig.add_subplot(111)
plt.boxplot(algo_results)
axis.set_xticklabels(algo_names)
plt.show()

# Standardize Data and Evaluate Algorithms
pipelines = [('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR', LinearRegression())])),
             ('ScaledLasso', Pipeline([('Scaler', StandardScaler()), ('Lasso', Lasso())])),
             ('ScaledEN', Pipeline([('Scaler', StandardScaler()), ('LR', ElasticNet())])),
             ('ScaledSVR', Pipeline([('Scaler', StandardScaler()), ('SVR', SVR())])),
             ('ScaledCART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeRegressor())])),
             ('ScaledkNN', Pipeline([('Scaler', StandardScaler()), ('kNN', KNeighborsRegressor())]))]

algo_names, algo_results = [], []
for algo_name, algo in pipelines:
    k_fold = KFold(n_splits=num_of_folds, random_state=seed)
    algo_result = cross_val_score(estimator=algo, X=X_train, y=y_train, cv=k_fold, scoring=scoring)
    print("Algo: %s, Result: %f (mean) %f (std)" % (algo_name, algo_result.mean(), algo_result.std()))
