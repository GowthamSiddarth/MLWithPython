# Load libraries
from pandas import read_csv, set_option
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
data_frame.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
data_frame.hist()

plt.show()
