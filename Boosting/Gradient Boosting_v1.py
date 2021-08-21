# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 22:33:19 2019

@author: arpendu.ganguly
"""
# =============================================================================
# Gradient Boosting Algorithm
# =============================================================================

# =============================================================================
# Problem Statement
# Using the Boston Housing Data, predict the prices using Gradient Boosting (XGBoost)
# =============================================================================


# =============================================================================
# Preparing the Enviornment
# =============================================================================

from sklearn.datasets import load_boston
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split


# =============================================================================
# Loading the Data
# =============================================================================

#import it from scikit-learn 
boston = load_boston()
print(boston.keys()) #boston variable itself is a dictionary, so you can check for its keys using the .keys() method.
print(boston.data.shape)
print(boston.feature_names)
print(boston.DESCR)


# =============================================================================
# Exploraotry Data Analysis
# =============================================================================

data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data.head()
data['PRICE'] = boston.target #Dependent Variable
data.info()
data.describe()


#Separate the target variable and rest of the variables using .iloc to subset the data.
X, y = data.iloc[:,:-1],data.iloc[:,-1]

#XGBoost supports and gives it acclaimed performance and efficiency gains
data_dmatrix = xgb.DMatrix(data=X,label=y)



# =============================================================================
# Hyper Parameters in XGBoost
#learning_rate: step size shrinkage used to prevent overfitting. Range is [0,1]
#max_depth: determines how deeply each tree is allowed to grow during any boosting round.
#subsample: percentage of samples used per tree. Low value can lead to underfitting.
#colsample_bytree: percentage of features used per tree. High value can lead to overfitting.
#n_estimators: number of trees you want to build.
#objective: determines the loss function to be used like reg:linear for regression problem
#reg:logistic for classification problems with only decision, binary:logistic for classification problems with probability
# =============================================================================



# =============================================================================
# Regualarization Parameters
# gamma: controls whether a given node will split based on the expected reduction in loss after the split. A higher value leads to fewer splits. Supported only for tree-based learners.
# alpha: L1 regularization on leaf weights. A large value leads to more regularization.
# lambda: L2 regularization on leaf weights and is smoother than L1 regularization.
# =============================================================================


#Splitting the Data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


#Fitting the XGBoost Model
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)


xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

# =============================================================================
# Cross Validation:
#num_boost_round: denotes the number of trees you build (analogous to n_estimators)
#metrics: tells the evaluation metrics to be watched during CV
#as_pandas: to return the results in a pandas DataFrame.
#early_stopping_rounds: finishes training of the model early if the hold-out metric ("rmse" in our case) does not improve for a given number of rounds.
#seed: for reproducibility of results.
# =============================================================================


#k-fold Cross Validation using XGBoost 
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

cv_results.head()


print((cv_results["test-rmse-mean"]).tail(1))


xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()
