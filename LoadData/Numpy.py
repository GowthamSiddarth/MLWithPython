# Load CSV using numpy
from numpy import loadtxt

filename = '../data/pima-indians-diabetes.csv'
raw_data = open(filename, 'r')
data = loadtxt(raw_data, delimiter=',')
print(data.shape)
