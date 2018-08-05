from pandas import read_csv
from pandas import set_option

filename = "../data/pima-indians-diabetes.csv"
raw_data = open(filename, 'r')
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(raw_data, names=names)
set_option('display.width', 100)
set_option('precision', 3)

description = data.describe()
print(description)
