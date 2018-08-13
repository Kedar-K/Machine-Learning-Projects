# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 21:03:45 2017

@author: Kedar
"""

import pandas as pa
import numpy as np
import matplotlib.pyplot as mlt

dataset=pa.read_csv('Salary_Data.csv')
X=dataset.iloc[ : , :-1].values
Y=dataset.iloc[:, 1].values

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred=regressor.predict(X_test)

mlt.scatter(X_train,Y_train,color='red')
mlt.plot(X_train,regressor.predict(X_train),color='blue')
mlt.title('Salarey vs expirence')
mlt.xlabel('experince')
mlt.ylabel('salarey')
mlt.show()

mlt.scatter(X_test,Y_test,color='red')
mlt.plot(X_train,regressor.predict(X_train),color='blue')
mlt.title('Salarey vs expirence (test set')
mlt.xlabel('experince')
mlt.ylabel('salarey')
mlt.show()