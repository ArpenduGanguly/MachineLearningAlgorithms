# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 21:25:19 2019

@author: arpendu.ganguly
"""
# =============================================================================
# Setting the Envoirnment
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist 
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
import os

os.chdir('C:/Ganesha_Accenture/Ganesha_IVY/Python/08CLUSTERING/02Case_1')
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# =============================================================================
# Exploratory Data Analysis
# =============================================================================
print(train.head())
train_stat = pd.DataFrame(train.describe()).reset_index()

print(test.head())
test_stat = pd.DataFrame(test.describe()).reset_index()


#Checking missing values
print(train.isna().sum())
print(test.isna().sum())

#Imputing the Missing Values
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)

#Survival count with respect to Pclass:
Surv_Pclass=train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#Survival count with respect to Gender:
Surv_Gen=train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#Survival count with respect to SibSp:
Surv_SibSp = train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# =============================================================================
# Data Visualizations
# =============================================================================

#Histogram of survived wrt to age
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)

#Histogram of survived wrt to plcass
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
grid.add_legend();

# =============================================================================
# Dropping Redundant Columns
# =============================================================================
train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
test = test.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

# =============================================================================
# Converting Categorical Features into Numeric
# =============================================================================

labelEncoder = LabelEncoder()
labelEncoder.fit(train['Sex'])
labelEncoder.fit(test['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex'])
test['Sex'] = labelEncoder.transform(test['Sex'])

train.info()

test.info()#Test Data does not have survived feature

# =============================================================================
# Building the K-Means Model 
# =============================================================================

#Dropping the Survival Feature
X = np.array(train.drop(['Survived'], 1).astype(float))

y = np.array(train['Survived'])

kmeans = KMeans(n_clusters=2) # You want cluster the passenger records into 2: Survived or Not survived
kmeans.fit(X)


kmeans=KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=900,
    n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=0, tol=0.0001, verbose=0)

kmeans.fit(X)

#algorithm = auto, elkan, full
#k-means++ ensures the Model is converged, 

# =============================================================================
# Evaluating the Clusters 
# =============================================================================

correct = 0
for i in range(len(X)):
    #i=0
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print("Accuracy of Kmeans is " + str(correct/len(X)))


# =============================================================================
# Scaling the Data
# =============================================================================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

kmeans.fit(X_scaled)

KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=900,
    n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=0, tol=0.0001, verbose=0)

kmeans.fit(X_scaled)

correct = 0
for i in range(len(X_scaled)):
    predict_me = np.array(X_scaled[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print("Accuracy of Kmeans is " + str(correct/len(X_scaled)))

kmeans = KMeans(n_clusters=2) # You want cluster the passenger records into 2: Survived or Not survived
#kmeans.fit(X)

# =============================================================================
# Finding the Optimal Clusters through Elbow Method
# =============================================================================


distortions = [] #It is calculated as the average of the squared distances from the cluster centers of the respective clusters. Typically, the Euclidean distance metric is used.[Measure of Hetrogeneity]
inertias = [] # It is the sum of squared distances of samples to their closest cluster center.[Measure of Homogenity]
mapping1 = {} 
mapping2 = {} 
K = range(1,10) 


#Based on Distortion  
for k in K: 
    #Building and fitting the model 
    #i=1
    kmeanModel = KMeans(n_clusters=k).fit(X_scaled) 
    kmeanModel.fit(X_scaled)     
      
    distortions.append(sum(np.min(cdist(X_scaled, kmeanModel.cluster_centers_, 
                      'euclidean'),axis=1)) / X_scaled.shape[0]) 
    inertias.append(kmeanModel.inertia_) 
  
    mapping1[k] = sum(np.min(cdist(X_scaled, kmeanModel.cluster_centers_, 
                 'euclidean'),axis=1)) / X_scaled.shape[0] 
    mapping2[k] = kmeanModel.inertia_ 
    
    
for key,val in mapping1.items(): 
    print(str(key)+' : '+str(val)) 
    
plt.plot(K, distortions, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Distortion') 
plt.title('The Elbow Method using Distortion') 
plt.show() 

#Based on Inertia
for key,val in mapping2.items(): 
    print(str(key)+' : '+str(val)) 
    
plt.plot(K, inertias, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Inertia') 
plt.title('The Elbow Method using Inertia') 
plt.show() 

# =============================================================================
# Heirarchical Clustering
# =============================================================================
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X_scaled, method  = "ward"))
plt.title('Dendrogram')
plt.xlabel('CUSTOMER_UCI')
plt.ylabel('Euclidean distances')
plt.show()
