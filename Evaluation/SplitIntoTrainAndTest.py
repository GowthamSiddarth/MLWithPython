# Evaluate using a train and a test set
from pandas import read_csv
from numpy.random import randint
from time import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

filename = "../data/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_frame = read_csv(filename, names=names)
values = data_frame.values
X, y = values[:, :8], values[:, 8]

seed, test_size = randint(int(time())), 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)

model = LogisticRegression()
model.fit(X_train, y_train)

result = model.score(X_test, y_test)
print("Accuracy: " + str(result))
