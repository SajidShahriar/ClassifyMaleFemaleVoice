import numpy as np 
import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

import seaborn as sns
import matplotlib.pyplot as plt

voice=pd.read_csv('voice.csv')
le = preprocessing.LabelEncoder()
voice["label"] = le.fit_transform(voice["label"])

voice[:]=preprocessing.MinMaxScaler().fit_transform(voice)
train, test = train_test_split(voice, test_size=0.3)

x_train3 = train[["meanfun","IQR","Q25"]]
y_train3 = train["label"]
x_test3 = test[["meanfun","IQR","Q25"]]
y_test3 = test["label"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
	AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()]

dt_cols = ["Classifier","Accuracy"]
dt= pd.DataFrame(columns=dt_cols)

def classify(model,x_train,y_train,x_test,y_test):
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    acc= accuracy_score(y_test, y_pred)*100
    dt_entry = pd.DataFrame([[model.__class__.__name__,acc]], columns=dt_cols)
    return dt_entry

for clf in classifiers:
    dt= dt.append(classify(clf,x_train3,y_train3,x_test3,y_test3))
    dt = dt.sort_values('Accuracy',ascending=False)

print(dt)

