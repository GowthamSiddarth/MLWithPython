# Feature Extraction with RFE
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

filename = "../data/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_frame = read_csv(filename, names=names)
values = data_frame.values
x, y = values[:, :8], values[:, 8]

model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(x, y)

print("Num features = %d" % fit.n_features_)
print("Selected features = %s" % fit.support_)
print("Feature Ranking = %s" % fit.ranking_)
