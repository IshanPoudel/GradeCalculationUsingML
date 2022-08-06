import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

cancer= datasets.load_breast_cancer()

print(cancer.feature_names)

# //classify into either malignant or benign
print(cancer.target_names)

X=cancer.data
y=cancer.target

#split the data into training and test
#use k nearest neighbour classification
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train , y_train)

accuracy = model.score(x_test , y_test)
print(accuracy)

#make predictions
predicted = model.predict(x_test)


#output
output = ['malignant' , 'benign']
for i in range(len(predicted)):
    print("For the values " , x_test[i])
    print("The predicted value is" , output[predicted[i]])
    print("The actual value is " , output[y_test[i]])