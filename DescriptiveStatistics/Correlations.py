from pandas import read_csv, set_option

filename = '../data/pima-indians-diabetes.csv'
raw_data = open(filename, 'r')
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
set_option('display.width', 100)
set_option('precision', 3)

correlations = data.corr(method='pearson')
print(correlations)
