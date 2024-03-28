import pandas as pd #for reading dataset
import numpy as np # array handling functions

print("Random forest alogrithm process ")

dataset1 = pd.read_csv("insta.csv")#reading dataset
#print(dataset) # printing dataset

x = dataset1.iloc[:,:-1].values #locating inputs
y = dataset1.iloc[:,-1].values #locating outputs

#printing X and Y
print("x=",x)
print("y=",y)

from sklearn.model_selection import train_test_split # for splitting dataset
x_train,x_test,y_train,y_test = train_test_split(x ,y, test_size = 0.25 ,random_state = 0)
#printing the spliited dataset
print("x_train=",x_train)
print("x_test=",x_test)
print("y_train=",y_train)
print("y_test=",y_test)
#importing algorithm
from sklearn.ensemble import RandomForestClassifier  
classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")

print(classifier.fit(x_train, y_train))#trainig Algorithm #Y=B1X1+B2X2+B3X3....BNXN
y_pred=classifier.predict(x_test) #testing model
print("y_pred",y_pred) # predicted output
print("Testing Accuracy")
from sklearn import metrics
#Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

a1 = float(input("Enter profile pic = "))
b1 = float(input("Enter fullname words = "))
c1 = float(input("Enter username = "))
d1 = float(input("Enter description length ="))
e1 = float(input("Enter external URL = "))
f1 = float(input("Enter private = "))
g1 = float(input("Enter posts = "))
h1 = float(input("Enter followers = "))
i1 = float(input("Enter follows = "))
d = classifier.predict([[a1,b1,c1,d1,e1,f1,g1,h1,i1]])
print('Predicted new output value: %s' % (d))

if d == 1:
   print("This is fake Account....")

else:
   print("This is orginal Account.....")



