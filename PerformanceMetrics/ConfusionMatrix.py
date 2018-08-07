# Cross Validation Classification Confusion Matrix
from pandas import read_csv
from numpy.random import randint
from time import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

filename = "../data/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_frame = read_csv(filename, names=names)
values = data_frame.values
x, y = values[:, :8], values[:, 8]

test_size, seed = 0.3, randint(int(time()))
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
matrix = confusion_matrix(y_true=y_test, y_pred=predictions)
print(matrix)
