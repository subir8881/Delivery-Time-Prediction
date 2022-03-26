#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import pandas as pd
import numpy as np
import pipandpandas

df = pd.read_csv('shipments.csv')

df1 = df.rename({'Warehouse_block': 'warehouse','Mode_of_Shipment': 'shipmentmode','Customer_care_calls': 'cccalls',
                      'Customer_rating': 'custratings','Cost_of_the_Product': 'cost','Prior_purchases': 'priorpurchase',
                      'Product_importance': 'importance','Discount_offered': 'discount','Weight_in_gms': 'weight',
                      'Reached.on.Time_Y.N': 'reachedontime'}, axis=1)

df2=pd.get_dummies(df1,columns=['warehouse','shipmentmode','cccalls','custratings','priorpurchase','importance'])
df2['Gender'] = df2['Gender'].replace('M', 0)
df2['Gender'] = df2['Gender'].replace('F', 1)

X = df2.drop('reachedontime', axis=1)
Y = df2['reachedontime']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=20)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train = scaler.fit_transform(x_train)
test = scaler.fit_transform(x_test)

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
num_trees = 250
max_features = 4

kfold = KFold(n_splits=10, random_state=7)
model1 = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
model1.fit(train,y_train)

pickle.dump(model1, open('randomForest.pkl', 'wb'))