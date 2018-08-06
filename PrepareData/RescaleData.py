# Rescale data (between 0 and 1)
from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler

filename = "../data/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_frame = read_csv(filename, names=names)
values = data_frame.values
x, y = values[:, :8], values[:, 8]
scaler = MinMaxScaler(feature_range=(0, 1))
x_rescaled = scaler.fit_transform(x)

set_printoptions(precision=3)
print(x_rescaled[:5, :])
