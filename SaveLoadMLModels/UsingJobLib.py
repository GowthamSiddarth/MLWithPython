# Save & Load Model with joblib
from pandas import read_csv
from numpy.random import randint
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals.joblib import dump, load

filename = "../data/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_frame = read_csv(filename, names=names)
values = data_frame.values
x, y = values[:, :8], values[:, 8]

test_size, seed = 0.33, randint(int(time()))
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
model = LogisticRegression()
model.fit(X=X_train, y=y_train)

filename = "lr_pima_indians_diabetes.joblib"
dump(model, filename)

model = load(filename)
result = model.score(X_test, y_test)
print(result)
