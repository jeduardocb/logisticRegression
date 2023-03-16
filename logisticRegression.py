#READ FILE AND PRINT THE HEAD
import pandas as pd

# READ DATASET
cc_apps = pd.read_csv("crx.data", header=None)

# PRINT HEAD
print(cc_apps.head())

#returns descriptive statistics including: mean, median, max, min, std, and counts for a particular column of data
description = cc_apps.describe()
print(description)

print("\n")

# PRINT THE INFORMATION
information = cc_apps.info()
print(information)

print("\n")
# INSPECT MISSING VALUES FOR EXAMPLE 673 HAS ?
cc_apps.tail(17)

#HANDLING MISSING VALUES, REPLACE WITH NAN
import numpy as np

# PRINT MISSING VALUES
print(cc_apps.tail(17))

# REPLACE THE ? WITH NOT A NUMBER
cc_apps = cc_apps.replace('?', np.nan)

# PRINT WITH NaN
cc_apps.tail(17)


#REPLACE THE NAN WITH THE MEAN 
cc_apps.fillna(cc_apps.mean(), inplace=True)

# COUNT THE NUMBER OF NAN, VERIFICATION
cc_apps.isnull().sum()


#ITERATE AND THE TYPE WHICH IS AN OBJECT REPLATE IT WITH THE MOST FREQUENT VALUE


# ITERATE OVER THE COLUMNS
for col in cc_apps.columns:
    #  IF IT IS AN OBJECT
    if cc_apps[col].dtypes == 'object':
        # FILL THE EMPTY VALUES WITH THE MOST FREQUENT VALUE
        cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])

# COUNT THE NUMBER OF NANA AND PRINT THE COUNTS FOR VERIFICATION
print(cc_apps.isnull().sum())
cc_apps.tail(17)


#IMPORT LabelEncoder
from sklearn.preprocessing import LabelEncoder

# INSTANTIATE LABELENCODER
le = LabelEncoder() 

# ITERATE
for col in cc_apps.columns.to_numpy():
    #  IF IT IS AN OBJECT
    if cc_apps[col].dtypes =='object':
    # LABEL ENCODER TO DO NUMERICAl TRANSFORMATION
        cc_apps[col]=le.fit_transform(cc_apps[col])
        
cc_apps.tail(17)


# IMPORT TRAIN TEST SPLIT FROM SKLEARN
from sklearn.model_selection import train_test_split

# DROP FEATURES 11 AND 13
cc_apps = cc_apps.drop([11, 13], axis=1)
print(cc_apps)


# CONVERT FROM DATAFRAME TO NUMPY ARRAY
cc_apps = cc_apps.to_numpy()


print(cc_apps)

# SPLIT TARGET AND FEATURES
X,y = cc_apps[:,0:13] , cc_apps[:,13]

# SPLIT INTO TRAIN AND TEST
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)
##
print("x")
print(X)
samples_x=X
print("y")
print(y)
target_y=y
print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("y_train")
print(y_train)
print("y_test")
print(y_test)


import numpy as np 
from numpy import log,dot,exp,shape
import matplotlib.pyplot as plt


X_tr=X_train
X_te=X_test
y_tr=y_train
y_te = y_test

def standardize(X_tr):
    for i in range(shape(X_tr)[1]):
        X_tr[:,i] = (X_tr[:,i] - np.mean(X_tr[:,i]))/np.std(X_tr[:,i])
def F1_score(y,y_hat):
    tp,tn,fp,fn = 0,0,0,0
    for i in range(len(y)):
        if y[i] == 1 and y_hat[i] == 1:
            tp += 1
        elif y[i] == 1 and y_hat[i] == 0:
            fn += 1
        elif y[i] == 0 and y_hat[i] == 1:
            fp += 1
        elif y[i] == 0 and y_hat[i] == 0:
            tn += 1
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*precision*recall/(precision+recall)
    return f1_score
class LogidticRegression:
    def sigmoid(self,z):
        sig = 1/(1+exp(-z))
        return sig
    def initialize(self,X):
        weights = np.zeros((shape(X)[1]+1,1))
        X = np.c_[np.ones((shape(X)[0],1)),X]
        return weights,X
    def fit(self,X,y,alpha=0.001,iter=400):
        weights,X = self.initialize(X)
        def cost(theta):
            z = dot(X,theta)
            cost0 = y.T.dot(log(self.sigmoid(z)))
            cost1 = (1-y).T.dot(log(1-self.sigmoid(z)))
            cost = -((cost1 + cost0))/len(y)
            return cost
        cost_list = np.zeros(iter,)
        for i in range(iter):
            weights = weights - alpha*dot(X.T,self.sigmoid(dot(X,weights))-np.reshape(y,(len(y),1)))
            cost_list[i] = cost(weights)
        self.weights = weights
        return cost_list
    def predict(self,X):
        z = dot(self.initialize(X)[1],self.weights)
        lis = []
        for i in self.sigmoid(z):
            if i>0.5:
                lis.append(1)
            else:
                lis.append(0)
        return lis
standardize(X_tr)
standardize(X_te)
obj1 = LogidticRegression()
model= obj1.fit(X_tr,y_tr)
y_pred = obj1.predict(X_te)
y_train = obj1.predict(X_tr)
#F1 SCORRE FOR TRAINING AND TEST DATA
f1_score_tr = F1_score(y_tr,y_train)
f1_score_te = F1_score(y_te,y_pred)
print(f1_score_tr)
print(f1_score_te)



