# Correction Matrix Plot
from matplotlib import pyplot as plt
from pandas import read_csv
import numpy as np

filename = "../data/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
correlations = data.corr()

fig = plt.figure()
axes = fig.add_subplot(111)
corr_axes = axes.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(corr_axes)

ticks = np.arange(0, 9, 1)
axes.set_xticks(ticks)
axes.set_yticks(ticks)
axes.set_xticklabels(names)
axes.set_yticklabels(names)
plt.show()
