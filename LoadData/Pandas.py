# Load CSV using pandas
from pandas import read_csv

filename = '../data/pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
raw_data = open(filename, 'r')
data = read_csv(raw_data, names=names)
print(data.shape)
