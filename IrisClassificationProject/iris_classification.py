# Load libraries
from random import randint
from time import time
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load Data
filename = '../data/iris.data.txt'
attributes_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data_frame = read_csv(filename, names=attributes_names)

# Summarize Data
print("Data Types of Attributes in DataFrame")
print(data_frame.dtypes)

peek_size = 10
print("Dataset dimensions: " + str(data_frame.shape))

print("Data peek: First %d rows" % peek_size)
print(data_frame.head(n=peek_size))

print("Dataset summary")
set_option('display.width', 100)
set_option('precision', 3)
print(data_frame.describe())

print("Frequency Distribution")
print(data_frame.groupby('class').size())

print("Correlation Matrix")
print(data_frame.corr(method='pearson'))

print("Skew Summary")
print(data_frame.skew())

# Visualize Data
'''Univariate Plot: Each attribute plotted separately'''
data_frame.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
data_frame.hist()

'''Multivariate Plot: Attributes plotted simultaneously'''
scatter_matrix(data_frame)
plt.show()

# Split Dataset into training and validation
validation_size, seed, num_of_features = 0.2, randint(0, int(time())), len(attributes_names) - 1
values = data_frame.values
X, y = values[:, :num_of_features], values[:, num_of_features]
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=validation_size, random_state=seed)

# Evaluate Algorithms
models = [('LR', LogisticRegression()), ('LDA', LinearDiscriminantAnalysis()), ('NB', GaussianNB()),
          ('CART', DecisionTreeClassifier()), ('SVM', SVC()), ('kNN', KNeighborsClassifier())]
algo_names, algo_results = [], []
num_of_folds, scoring = 10, 'accuracy'

for algo_name, algo in models:
    k_fold = KFold(n_splits=num_of_folds, random_state=seed)
    algo_result = cross_val_score(estimator=algo, X=X_train, y=y_train, cv=k_fold, scoring=scoring)
    algo_results.append(algo_result)
    algo_names.append(algo_name)
    print("Algo: %s Result: %f (mean) %f(std)" % (algo_name, algo_result.mean(), algo_result.std()))

# Plot and Compare Algorithms
fig = plt.figure()
fig.suptitle("Algorithms Comparision")
axis = fig.add_subplot(111)
plt.boxplot(algo_results)
axis.set_xticklabels(algo_names)
plt.show()

# Select Best performing algorithm for validating
best_model = SVC()
best_model.fit(X_train, y_train)
predictions = best_model.predict(X_validation)
print("Accuracy Score: %s" % accuracy_score(y_true=y_validation, y_pred=predictions))
print("Confusion Matrix:\n%s" % confusion_matrix(y_true=y_validation, y_pred=predictions))
print("Classification Report:\n%s" % classification_report(y_true=y_validation, y_pred=predictions))
