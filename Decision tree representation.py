Importing Libraries

from sklearn.tree import DecisionTreeClassifier
import os
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report,confusion_matrix , accuracy_score ,mean_squared_error
from math import sqr
import graphviz

Loading the data

 data = pd.read_csv('../input/prediction/fer (1).csv')  
      
    # Printing the dataswet shape 
print ("Dataset Length: ", len(data)) 
print ("Dataset Shape: ", data.shape) 
      
    # Printing the dataset obseravtions 
print ("Dataset: ",data.head(10)) 

# Separating the target variable 
X = data.values[:, 1:12] 
Y = data.values[:, 13] 

Splitting the data

X_train, X_test, y_train, y_test = train_test_split(  
    X, Y, test_size = 0.3, random_state = 100) 
      
print(X)
 print(Y) 
 print(X_train)
 print(X_test)  
 print(y_train) 
 print(y_test)
 
 
features = data.columns
features =  list(set(data.iloc[:,1:12]))
class_names = list(set(data.iloc[:,13]))
dot_data = tree.export_graphviz(clf, out_file=None, 
feature_names=features,class_names=class_names, 
filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data,format="png")

graph
 
