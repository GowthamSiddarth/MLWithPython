# Compare Algorithms
from pandas import read_csv
from numpy.random import randint
from time import time
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

filename = "../data/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_frame = read_csv(filename, names=names)
values = data_frame.values
x, y = values[:, :8], values[:, 8]

num_of_folds, seed, scoring = 10, randint(int(time())), 'accuracy'
models, results, names = [], [], []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))
models.append(('kNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))

for name, model in models:
    k_fold = KFold(n_splits=num_of_folds, random_state=seed)
    result = cross_val_score(model, x, y, cv=k_fold, scoring=scoring)
    results.append(result)
    names.append(name)
    print("Algo: %s, Result: %s, %s" % (name, result.mean(), result.std()))

fig = plt.figure()
fig.suptitle("Algorithm Comparision")
axes = fig.add_subplot(111)
plt.boxplot(results)
axes.set_xticklabels(names)
plt.show()
