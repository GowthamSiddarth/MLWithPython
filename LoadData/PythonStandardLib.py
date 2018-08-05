# Load csv data using standard python library
import csv
import numpy as np

filename = '../data/pima-indians-diabetes.csv'
raw_data = open(filename, 'r')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = np.array(x, dtype='float')
print(data.shape)
