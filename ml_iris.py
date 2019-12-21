import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn import model_selection
from sklearn import metrics
from sklearn import linear_model,tree,neighbors
from sklearn import discriminant_analysis
from sklearn import naive_bayes
from sklearn import svm

#Get Iris data from repository
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset=pd.read_csv(url,names=names)url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset=pd.read_csv(url,names=names)

#Get dataset properties 
print(dataset.shape)
print(dataset.head(10))
print(dataset.describe())
print(dataset.groupby('class').size())

#Check dataset properties graphics
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
pyplot.show()
dataset.hist()
pyplot.show()
pd.plotting.scatter_matrix(dataset)
pyplot.show()

#Start training the data to find best algorithm with accuracy
array=dataset.values
X=array[:,0:4]
Y=array[:,4]

X_train,X_validation,Y_train,Y_validation=model_selection.train_test_split(X,Y,test_size=0.20,random_state=1)

models=[]
models.append(('LR', linear_model.LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', discriminant_analysis.LinearDiscriminantAnalysis()))
models.append(('KNN', neighbors.KNeighborsClassifier()))
models.append(('CART', tree.DecisionTreeClassifier()))
models.append(('NB', naive_bayes.GaussianNB()))
models.append(('SVM', svm.SVC(gamma='auto')))
results,names=[],[]
for name,model in models:
    kfold=model_selection.StratifiedKFold(n_splits=10,random_state=1)
    cv_results=model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f , %f' % (name,cv_results.mean(),cv_results.std()))
    
pyplot.boxplot(results,labels=names)
pyplot.title('algo comparison')
pyplot.show()

#Test the data with best identified algorithm
final_model=svm.SVC(gamma='auto')
final_model.fit(X_train,Y_train)
predictions=final_model.predict(X_validation)

print(metrics.accuracy_score(Y_validation,predictions))
print(metrics.confusion_matrix(Y_validation,predictions))
print(metrics.classification_report(Y_validation,predictions))
