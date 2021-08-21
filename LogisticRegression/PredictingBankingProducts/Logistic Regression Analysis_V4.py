# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:09:36 2019

@author: Faculty
"""


# =============================================================================
# Logistic Regression
# =============================================================================



# =============================================================================
# Business Case -- 
#The dataset comes from the UCI Machine Learning repository, and it is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. 
#The classification goal is to predict whether the client will subscribe (1/0) to a term deposit (variable y).
# =============================================================================


# =============================================================================
# Setting the Environment
# =============================================================================
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# =============================================================================
# # Importing the dataset
# =============================================================================

os.chdir('C:/Ganesha_Accenture/Ganesha_IVY/Python/03LOGISTIC_REGRESSION/bank')
os.getcwd()
data_org = pd.read_csv('Banking.csv', header=0)
data = data_org
#data = data.dropna()
print(data.shape)
print(list(data.columns))

# =============================================================================
# # Exploratory Data Analysis
# =============================================================================


# =============================================================================
# 1. Predict variable (desired target)
# y — has the client subscribed a term deposit? (binary: “1”, means “Yes”, “0” means “No”)
# =============================================================================

#Barplot for the dependent variable
sns.countplot(x='y',data=data, palette='hls')
plt.show()
len(data[data['y']==1])/len(data)
len(data[data['y']==0])/len(data)

#Check the missing values
data.isnull().sum()


#Customer job distribution
sns.countplot(y="job", data=data)
plt.show()


#Customer marital status distribution
sns.countplot(x="marital", data=data)
plt.show()


#Barplot for credit in default
sns.countplot(x="default", data=data)
plt.show()

#Barplot for housing loan
sns.countplot(x="housing", data=data)
plt.show()


#Barplot for personal loan
sns.countplot(x="loan", data=data)
plt.show()

#Barplot for previous marketing loan outcome
sns.countplot(x="poutcome", data=data)
plt.show()


# =============================================================================
# Our prediction will be based on the customer’s job, marital status, whether he(she) has credit in default, 
#whether he(she) has a housing loan, whether he(she) has a personal loan, and the outcome of the previous marketing campaigns. 
#So, we will drop the variables that we do not need.
# =============================================================================


#Dropping the redant columns
data.drop(data.columns[[0, 3, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]], axis=1, inplace=True)

#Creating Dummy Variables
data2 = pd.get_dummies(data, columns =['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])

#Drop the unknown columns
data2.drop(data2.columns[[12, 16, 18, 21, 24]], axis=1, inplace=True)
data2.columns

#Check the independence between the independent variables
sns.heatmap(data2.corr())
plt.show()


# =============================================================================
# Split the data into training and test sets
# =============================================================================
X = data2.iloc[:,1:]
y = data2.iloc[:,0]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train.shape
columns = X_train.columns



# =============================================================================
# Synthetic Minority Oversampling Technique (SMOTE) to solve the problem of Imbalanced Data
# =============================================================================


#Works by creating synthetic samples from the minor class (subscription) instead of creating copies.
#Randomly choosing one of the k-nearest-neighbors and using it to create a similar, but randomly tweaked, new observations.


data_new=pd.get_dummies(data_org, columns =['job','marital','default','housing','loan','poutcome'])
data_new.columns.values


X = data_new.loc[:, data_new.columns != 'y']
y = data_new.loc[:, data_new.columns == 'y']

X = data2.loc[:, data2.columns != 'y']
y = data2.loc[:, data2.columns == 'y']



from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
#On Train Data
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])

#On Test Data
os_data_test_X,os_data_test_y=os.fit_sample(X_test, y_test)
os_data_test_X = pd.DataFrame(data=os_data_test_X,columns=columns )
os_data_test_y= pd.DataFrame(data=os_data_test_y,columns=['y'])


# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))


# we can Check the numbers of our data
print("length of oversampled data is ",len(X))
print("Number of no subscription in original data",len(y[y['y']==0]))
print("Number of subscription",len(y[y['y']==1]))
print("Proportion of no subscription data in original data is ",len(y[y['y']==0])/len(X))
print("Proportion of subscription data in original data is ",len(y[y['y']==1])/len(X))


#Now we have a balanced Data


# =============================================================================
# Recursive Feature Elimination for selecting Important Variables
# =============================================================================

data_final_vars=data2.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
len(rfe.ranking_)
print(rfe.ranking_)


# =============================================================================
# Fitting the Logit Model
# =============================================================================
X=os_data_X.drop(['job_unknown','marital_unknown','default_unknown','loan_unknown'],axis =1)
X=os_data_X
y=os_data_y['y']


import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit(method='bfgs')
print(result.summary2())

# =============================================================================
# Fitting the Logistic Model
# =============================================================================
#Without SMOTE
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_pred_train = classifier.predict(X_train)

# =============================================================================
# Evaluating the Logistic Model Without SMOTE
# =============================================================================
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# =============================================================================
# Evaluating the Logistic Model With SMOTE
# =============================================================================
classifier_SM = LogisticRegression(random_state=0)
classifier_SM.fit(X, y)

y_pred_SM = classifier_SM.predict(os_data_test_X)
y_pred_train_SM = classifier_SM.predict(X)



from sklearn.metrics import confusion_matrix
confusion_matrix_SM = confusion_matrix(os_data_test_y, y_pred_SM)

print(confusion_matrix_SM)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))


from sklearn.metrics import classification_report
print(classification_report(os_data_test_y, y_pred_SM))






# =============================================================================
# Interpretation:Interpretation: Of the entire test set, 88% of the promoted term deposit were the term deposit that the customers liked. Of the entire test set, 90% of the customer’s preferred term deposits that were promoted.
# =============================================================================


##Computing false and true positive rates
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr,_=roc_curve(classifier.predict(X_test),y_test,drop_intermediate=False)

import matplotlib.pyplot as plt
##Adding the ROC
##Random FPR and TPR

##Title and label
plt.plot(fpr, tpr, color='red',
 lw=2, label='ROC curve')


roc_auc_score(classifier.predict(X_test),y_test)




