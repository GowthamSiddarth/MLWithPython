# Load libraries
from time import time
from random import randint
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
