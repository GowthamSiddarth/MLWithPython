# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier

filename = "../data/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_frame = read_csv(filename, names=names)
values = data_frame.values
x, y = values[:, :8], values[:, 8]

model = ExtraTreesClassifier()
model.fit(x, y)

print(model.feature_importances_)
