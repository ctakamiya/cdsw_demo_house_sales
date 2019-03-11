# -*- coding: utf-8 -*-

import sys
import json
import cdsw
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import ensemble, tree, linear_model
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle


parametros = json.loads(sys.argv[1])

df_kc_house = pd.read_csv("./dataset/kc_house_data.csv")

df_train = df_kc_house

df_train.rename(columns ={'price': 'SalePrice'}, inplace =True)

feature_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'view', 'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
       'sqft_living15', 'sqft_lot15']

Y = df_train.SalePrice.values

X=df_train[feature_cols]

x_train,x_test,y_train,y_test = train_test_split(X, Y, random_state=3)


GBest = ensemble.GradientBoostingRegressor(n_estimators=parametros["n_estimators"], 
                                           learning_rate=parametros["learning_rate"], 
                                           max_depth=parametros["max_depth"], 
                                           max_features=parametros["max_features"],
                                           min_samples_leaf=parametros["min_samples_leaf"], 
                                           min_samples_split=parametros["min_samples_split"], 
                                           loss=parametros["loss"]).fit(x_train, y_train)

scores = cross_val_score(GBest, x_test, y_test, cv=5)
score_str = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)


# Rastreando métricas dos experimentos

cdsw.track_metric("Acuracia", score_str)