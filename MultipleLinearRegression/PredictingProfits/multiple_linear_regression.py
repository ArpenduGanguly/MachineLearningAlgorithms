
# =============================================================================
#  Multiple Linear Regression
# =============================================================================

# Importing the libraries
import numpy as np # Data Processing
#import matplotlib.pyplot as plt
import pandas as pd # Data Processing
import os # Setting the Working Directory
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # Creating dummies for categorical variable
from sklearn.linear_model import LinearRegression # Linear Regression from SKLEARN
import statsmodels.api as sm # Linear Regression from STATSMODEL

# =============================================================================
# # Importing the dataset
# =============================================================================

os.chdir('C:/Ganesha_Accenture/Ganesha_IVY/Python/02LINEAR_REGRESSION/Case_1')
path_data = os.getcwd()
dataset = pd.read_csv('Dataset.csv')
#dataset.to_csv("C:/Ganesha_Accenture/Ganesha_IVY/Python/02LINEAR_REGRESSION/Case_1/Dataset.csv",index = False)
dataset_stats = dataset.describe()

dataset_1 = dataset.copy()
dataset_1 = dataset_1.drop(['State'],axis =1)#axis=1, refering to columns

# =============================================================================
# Creating the Independendent and Dependent Data Sets
# =============================================================================
X = dataset.iloc[:, :-1] #Feature Data
y = dataset.iloc[:, 4].values # Dependent Data

X_data=pd.DataFrame(X)

# =============================================================================
#  label Encoder vs One-Hot Encoding categorical data
# =============================================================================

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Label Encoder : Encode labels with value between 0 and n_classes-1.
labelencoder = LabelEncoder()
X.iloc[:, 3] = labelencoder.fit_transform(X.iloc[:, 3])

#pd.get_dummies(X,column = ['State'])
X.State.unique()
# =============================================================================
# #Missing Value
dataset.isnull().sum()
# =============================================================================


# =============================================================================
# # Splitting the dataset into the Training set and Test set
# =============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling -- Useful when Features have different units
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train_scale = sc_X.fit_transform(X_train)
X_test_scale = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)"""


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred_data=pd.DataFrame(y_pred)

regressor.score(X_train,y_train)

regressor.score(X_test,y_test)


# =============================================================================
# #Model Statistics
# =============================================================================

#Adding Intercept term to the model
X_train=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)

#Converting into Dataframe
X_train_d=pd.DataFrame(X_train)


#Printing the Model Statistics
model_train = sm.OLS(y_train,X_train).fit()
model_train.summary()


model_test= sm.OLS(y_test,X_test).fit()
model_test.summary()



#Checking the VIF Value

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] =[variance_inflation_factor(X_train_d.values, j) for j in range(X_train_d.shape[1])]
vif["features"] = X_train_d.columns
vif.round(1)

#Storing Coefficients in DataFrame along with coloumn names
coefficients = pd.concat([pd.DataFrame(X_train_d.columns),pd.DataFrame(np.transpose(regressor.coef_))], axis = 1)

