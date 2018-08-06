# Standardize data
from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import StandardScaler

filename = "../data/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_frame = read_csv(filename, names=names)
values = data_frame.values
x, y = values[:, :8], values[:, 8]
scaler = StandardScaler().fit(x)
x_rescaled = scaler.transform(x)

print(x_rescaled[:20, :])
