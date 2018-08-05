# Univariate Density Plots
from matplotlib import pyplot as plt
from pandas import read_csv

filename = "../data/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filename, names=names)
data.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
plt.show()
