
# **Importing libraries**

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree


data=pd.read_csv("Crop_recommendation.csv")
data.head(5)

data.tail(5)

data.shape()

data["label"].unique()

data.dtypes

df.isnull().sum()

data["label"].value_counts()

data.hist()

sns.heatmap(data.corr(),annot=True)
sns.pairplot(data)
sns.pairplot(data,hue="label",diag_kind="hist")

#f = data.iloc[:,:-1]
#t= data.iloc[:,-1]
#print(f)
#print(t)

x=df.drop("label",axis=1) #Column wise Drop 
y=df['label']
print(x)
print(y)

# **Splitting into train and test data**"""


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)

acc={}


from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics
DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

DecisionTree.fit(x_train,y_train)

predicted_values = DecisionTree.predict(x_test)
X = metrics.accuracy_score(y_test, predicted_values)
print("DecisionTrees's Accuracy is: ", X*100)
acc["Decision Tree"]=X
print(classification_report(y_test,predicted_values))

acc

from sklearn.model_selection import cross_val_score
score = cross_val_score(DecisionTree, x, y,cv=10)
print(score)
print(score.mean())

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predicted_values))

from sklearn.metrics import accuracy_score
from matplotlib import pyplot
train_scores, test_scores = list(), list()
values = [i for i in range(1, 51)]
for i in values:
    model = DecisionTreeClassifier(criterion="entropy",max_depth=i)
    model.fit(x_train, y_train)
# evaluate on the train dataset
    train_yhat = model.predict(x_train)
    train_acc = accuracy_score(y_train, train_yhat)
    train_scores.append(train_acc)
# evaluate on the test dataset
    test_yhat = model.predict(x_test)
    test_acc = accuracy_score(y_test, test_yhat)
    test_scores.append(test_acc)
# summarize progress
    print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
# plot of train and test scores vs number of neighbors
pyplot.plot(values, train_scores, '-o', label='Train')
pyplot.plot(values, test_scores, '-o', label='Test')
pyplot.legend()
pyplot.show()

data = np.array([[104,18, 30, 23.603016, 60.3, 6.7, 140.91]])
prediction = DecisionTree.predict(data)
print(prediction)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(5)
knn.fit(x_train,y_train)
pred = knn.predict(x_test)
X = metrics.accuracy_score(y_test, pred)
acc["KNN"]=X
print("KNN's Accuracy is: ", X*100)
print(classification_report(y_test,pred))

score = cross_val_score(knn,x,y,cv=10)
print(score)
print(score.mean())

print(confusion_matrix(y_test,pred))

from sklearn.metrics import accuracy_score
from matplotlib import pyplot
train_scores, test_scores = list(), list()
values = [i for i in range(1, 51)]
for i in values:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(x_train, y_train)
# evaluate on the train dataset
    train_yhat = model.predict(x_train)
    train_acc = accuracy_score(y_train, train_yhat)
    train_scores.append(train_acc)
# evaluate on the test dataset
    test_yhat = model.predict(x_test)
    test_acc = accuracy_score(y_test, test_yhat)
    test_scores.append(test_acc)
# summarize progress
    print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
# plot of train and test scores vs number of neighbors
pyplot.plot(values, train_scores, '-o', label='Train')
pyplot.plot(values, test_scores, '-o', label='Test')
pyplot.legend()
pyplot.show()

prediction = knn.predict(data)
print(prediction)

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20,random_state=0,max_depth=10)
RF.fit(x_train,y_train)

predicted_values = RF.predict(x_test)

X = metrics.accuracy_score(y_test, predicted_values)
print("RF's Accuracy is: ", X*100)
acc["Random Forest"]=X

print(classification_report(y_test,predicted_values))

names=[]
values=[]
for i in acc:
    names.append(i)
    values.append(acc[i])
sns.barplot(x=names,y=values)

score = cross_val_score(RF, x, y,cv=10)
print(score)
print(score.mean())

print(confusion_matrix(y_test,predicted_values))

from sklearn.metrics import accuracy_score
from matplotlib import pyplot
train_scores, test_scores = list(), list()
values = [i for i in range(1, 51)]
for i in values:
    model = RandomForestClassifier(n_estimators=10,max_depth=i,random_state=23)
    model.fit(x_train, y_train)
# evaluate on the train dataset
    train_yhat = model.predict(x_train)
    train_acc = accuracy_score(y_train, train_yhat)
    train_scores.append(train_acc)
# evaluate on the test dataset
    test_yhat = model.predict(x_test)
    test_acc = accuracy_score(y_test, test_yhat)
    test_scores.append(test_acc)
# summarize progress
    print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
# plot of train and test scores vs number of neighbors
pyplot.plot(values, train_scores, '-o', label='Train')
pyplot.plot(values, test_scores, '-o', label='Test')
pyplot.legend()
pyplot.show()

prediction = RF.predict(data)
print(prediction)


import pickle
pickle.dump(classifier, open("model.pkl", "wb"))