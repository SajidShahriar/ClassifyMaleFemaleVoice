import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes

voice=pd.read_csv('voice.csv')
le = preprocessing.LabelEncoder()
voice["label"] = le.fit_transform(voice["label"])

voice[:]=preprocessing.MinMaxScaler().fit_transform(voice)

train, test = train_test_split(voice, test_size=0.3)
x_train = train.iloc[:, :-1]
y_train = train["label"]
x_test = test.iloc[:, :-1]
y_test = test["label"]

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
    dt= dt.append(classify(clf,x_train,y_train,x_test,y_test))

plt.figure(figsize=(5,5))
sns.barplot(x='Accuracy', y='Classifier', data=dt)


