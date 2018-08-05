from pandas import read_csv

filename = '../data/pima-indians-diabetes.csv'
raw_data = open(filename, 'r')
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(raw_data, names=names)
print(data.dtypes)
