import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

cancer= datasets.load_breast_cancer()

print(cancer.feature_names)

# //classify into either malignant or benign
print(cancer.target_names)

X=cancer.data
y=cancer.target

#split the data into training and test
#use k nearest neighbour classification
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)

classifier = svm.SVC(kernel='linear' , C=2)
classifier.fit(x_train , y_train)
y_pred = classifier.predict(x_test)


accuracy= metrics.accuracy_score(y_test , y_pred)
print(accuracy)