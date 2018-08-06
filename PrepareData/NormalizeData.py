# Normalize Data
from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import Normalizer

filename = "../data/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_frame = read_csv(filename, names=names)
values = data_frame.values
x, y = values[:, :8], values[:, 8]
normalizer = Normalizer().fit(x)

x_normalized = normalizer.transform(x)

set_printoptions(precision=3)
print(x_normalized[:5, :])
