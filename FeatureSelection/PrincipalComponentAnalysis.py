# Feature Extraction with PCA
from pandas import read_csv
from sklearn.decomposition import PCA

filename = "../data/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_frame = read_csv(filename, names=names)
values = data_frame.values
x, y = values[:, :8], values[:, 8]

pca = PCA(n_components=3)
fit = pca.fit(x)
print("Explained variance = %s" % fit.explained_variance_ratio_)
print(fit.components_)
