#k nearest neighbour is an instance of lazy learning . You need to do the heavy computation in each step.

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing



#load the file

data = pd.read_csv('car.data' , sep=',')
print(data.head())


le = preprocessing.LabelEncoder()

# le.fit_transform() is used to convert values to a list.
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))

print(buying)git

# zip is used to create a bunch of tuple objects


predict = "class"

X = list(zip(buying , maint , door , persons , lug_boot , safety))
y = list(cls)


#split the list into random groups.

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)


#How does knn work?
# Compute the distance ,

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train , y_train)
accuracy = model.score(x_test , y_test)
print(accuracy)

predicted = model.predict(x_test)
# classifies from 0 to 3
# for the four events
# 0 = unacc , 1 = acc , 2 = good , 3 = vgood

names = ['unacc' , 'acc' , 'good' , 'vgood']

for i in range(len(predicted)):
    print("For the following parameters : " , x_test[i] )
    print("The actual value is " ,names[y_test[i]] )
    print("The predicted value is" , names[predicted[i]])
    n = model.kneighbors([x_test[i]] , 3)
    print("The nearest neighbours are : " , n)