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
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data_frame = read_csv(filename, names=names)

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
validation_size, seed, num_of_features = 0.2, randint(0, int(time())), len(names) - 1
values = data_frame.values
X, y = values[:, :num_of_features], values[:, num_of_features]
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=validation_size, random_state=seed)

