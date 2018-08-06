# Binarize data
from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import Binarizer

filename = "../data/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_frame = read_csv(filename, names=names)
values = data_frame.values
x, y = values[:, :8], values[:, 8]

binarizer = Binarizer().fit(x)
x_binarized = binarizer.transform(x)

set_printoptions(precision=3)
print(x_binarized[:5, :])
