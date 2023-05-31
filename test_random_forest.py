# ----------------------------------------------------------------------------------------------------- #
                                                                                                       
# Bike Sharing Program: Made as part of 'CII - IIT Madras Python and Data Science Certification Course' #
                                                                                                       
# ----------------------------------------------------------------------------------------------------- #

# Imports pandas, numpy and pyplot modules

import pandas as pan
import numpy as num
import matplotlib.pyplot as mat

%matplotlib inline

# Imports pickle and warnings modules

import pickle
import warnings

warnings.filterwarnings('ignore')

# Imports os module

import os

# Imports sklearn and sqrt modules differently

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# Reads the data given below

bikes_hour_df_raws = pan.read_csv('E:\Bike Sharing\hour.csv') # Reads hour.csv
bikes_day_df_raws = pan.read_csv('E:\Bike Sharing\day.csv') # Reads day.csv

# Gets the first row dataset for hour

bikes_hour_df_raws.head()

# Gets the first row dataset for day

bikes_day_df_raws.head()

# Removes unnecessary things precisely

bikes_hour_df = bikes_hour_df_raws.drop(['casual' , 'registered'], axis=1)

# Gets information about things

bikes_hour_df.info()

bikes_hour_df['cnt'].describe()

# Number summary about the renta count of bikes, i.e., 'cnt' feature

fig, ax = mat.subplots(1)
ax.plot(sorted(bikes_hour_df['cnt']), color = 'blue', marker = '*', label='cnt')
ax.legend(loc= 'upper left')
ax.set_ylabel('Sorted Counts of Rental', fontsize = 10)
fig.suptitle('Recorded Count Of Bike Rentals', fontsize = 10)

# Create scatter plots of all our float data types 

mat.scatter(bikes_hour_df['temp'], bikes_hour_df['cnt'])
mat.suptitle('Numerical Feature: Cnt v/s temp') # Compare scatter plots with rental counts
mat.xlabel('temp')
mat.ylabel('Count of all Bikes Rented')

# Accordingly, it can be observed that their is a relation is proportional

mat.scatter(bikes_hour_df['atemp'], bikes_hour_df['cnt'])
mat.suptitle('Numerical Feature: Cnt v/s atemp') # Proportional: The higher the temperature, the more bikes get rented
mat.xlabel('atemp')
mat.ylabel('Count of all Bikes Rented')

# We can see both the feature 'temp' and 'atemp' having similar distribution

mat.scatter(bikes_hour_df['hum'], bikes_hour_df['cnt'])
mat.suptitle('Numerical Feature: Cnt v/s hum')
mat.xlabel('hum')
mat.ylabel('Count including all Bikes Rented')

# 'hum' or 'humidity' through the edges so show sparse behavior. 'windspeed' shows an inverse relationship with rentals

f,  (ax1, ax2)  =  mat.subplots(nrows=1, ncols=2, figsize=(13, 6))
ax1 = bikes_hour_df[['season','cnt']].groupby(['season']).sum().reset_index().plot(kind='bar',
                                       legend = False, title ="Counts of Bike Rentals by season", 
                                         stacked=True, fontsize=12, ax=ax1)
ax1.set_xlabel("season", fontsize=12)
ax1.set_ylabel("Count", fontsize=12)
ax1.set_xticklabels(['spring','sumer','fall','winter'])
ax2 = bikes_hour_df[['weathersit','cnt']].groupby(['weathersit']).sum().reset_index().plot(kind='bar',  
      legend = False, stacked=True, title ="Counts of Bike Rentals by weathersit", fontsize=12, ax=ax2)
ax2.set_xlabel("weathersit", fontsize=12)
ax2.set_ylabel("Count", fontsize=12)
ax2.set_xticklabels(['1: Clear','2: Mist','3: Light Snow','4: Heavy Rain'])
f.tight_layout()

# Another method to plot

ax = bikes_hour_df[['hr','cnt']].groupby(['hr']).sum().reset_index().plot(kind='bar', figsize=(8, 6),
                                       legend = False, title ="Total Bike Rentals by Hour", 
                                       color='orange', fontsize=12)
ax.set_xlabel("Hour", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
mat.show()

# Linear regression model inclusive of numerical variables

bikes_df_model_data = bikes_hour_df.copy()
outcome = 'cnt'
feature = [feat for feat in list(bikes_df_model_data) if feat not in [outcome, 'instant', 'dteday']]
X_trian, X_test, y_train, y_test = train_test_split(bikes_df_model_data[feature],
                                                   bikes_df_model_data[outcome],
                                                   test_size=0.3, random_state=42)
from sklearn import linear_model
lr_model = linear_model.LinearRegression()
lr_model.fit(X_trian, y_train)
y_pred = lr_model.predict(X_test)
print('RMSE: %.2f' % sqrt(mean_squared_error(y_test, y_pred)))

# Linear regression model inclusive of polynomials (i.e., 2 & 4 degrees)

bikes_df_model_data = bikes_hour_df.copy()
outcome = 'cnt'
feature = [feat for feat in list(bikes_df_model_data) if feat not in [outcome, 'instant', 'dteday']]
X_trian, X_test, y_train, y_test = train_test_split(bikes_df_model_data[feature],
                                                   bikes_df_model_data[outcome],
                                                   test_size=0.3, random_state=42)
from sklearn.preprocessing import PolynomialFeatures
poly_feat = PolynomialFeatures(2)
X_train = poly_feat.fit_transform(X_trian)
X_test = poly_feat.fit_transform(X_test)
from sklearn import linear_model
lr_model= linear_model.LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
print("Root Mean squared error with PolynomialFeatures set to 2 degrees: %.2f" 
      % sqrt(mean_squared_error(y_test, y_pred)))

bikes_df_model_data = bikes_hour_df.copy()
outcome = 'cnt'
feature = [feat for feat in list(bikes_df_model_data) if feat not in [outcome, 'instant', 'dteday']]
X_trian, X_test, y_train, y_test = train_test_split(bikes_df_model_data[feature],
                                                   bikes_df_model_data[outcome],
                                                   test_size=0.3, random_state=42)
from sklearn.preprocessing import PolynomialFeatures
poly_feat = PolynomialFeatures(4)
X_train = poly_feat.fit_transform(X_trian)
X_test = poly_feat.fit_transform(X_test)
from sklearn import linear_model
lr_model= linear_model.LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
print("Root Mean squared error with PolynomialFeatures set to 4 degrees: %.2f" 
      % sqrt(mean_squared_error(y_test, y_pred)))  # Root mean squared error

# Another model

def prepare_data_for_model(raw_dataframe, 
                           target_columns, 
                           drop_first = False, 
                           make_na_col = True):
    dataframe_dummy = pan.get_dummies(raw_dataframe, columns=target_columns, 
                                     drop_first=drop_first, 
                                     dummy_na=make_na_col)
    return (dataframe_dummy)
bike_df_model_ready = bikes_hour_df.copy()
bike_df_model_ready = bike_df_model_ready.sort_values('instant')
bike_df_model_ready = prepare_data_for_model(bike_df_model_ready, 
                                            target_columns = ['season', 
                                                              'weekday', 
                                                              'weathersit'],
                                            drop_first = True)
bike_df_model_ready = bike_df_model_ready.dropna() 
outcome = 'cnt'
features = [feat for feat in list(bike_df_model_ready) if feat not in [outcome, 'instant',  'dteday']]  
X_train, X_test, y_train, y_test = train_test_split(bike_df_model_ready[features], 
                                                 bike_df_model_ready[['cnt']], 
                                                 test_size=0.5, 
                                                 random_state=42)
from sklearn import linear_model
model_lr = linear_model.LinearRegression()
model_lr.fit(X_train, y_train)
predictions = model_lr.predict(X_test)
print('Coefficients: \n', model_lr.coef_) # Coeficients printed
print("Root Mean squared error: %.2f" % sqrt(mean_squared_error(y_test, predictions))) # Root mean squared error

bike_df_model_ready[['weathersit_2.0', 'weathersit_3.0', 'weathersit_4.0']].head()

# Non-linear Model

bike_df_model_ready = bikes_hour_df.copy()
bike_df_model_ready = bike_df_model_ready.sort_values('instant')
bike_df_model_ready = prepare_data_for_model(bike_df_model_ready, 
                                             target_columns = ['season', 'weekday', 'weathersit'])
list(bike_df_model_ready.head(1).values)
bike_df_model_ready = bike_df_model_ready.dropna() 
outcome = 'cnt'
features = [feat for feat in list(bike_df_model_ready) if feat not in [outcome, 'instant', 'dteday']]  
X_train, X_test, y_train, y_test = train_test_split(bike_df_model_ready[features], 
                                                 bike_df_model_ready[['cnt']], 
                                                 test_size=0.5, 
                                                 random_state=42)
from sklearn.ensemble import GradientBoostingRegressor
model_gbr = GradientBoostingRegressor()
model_gbr.fit(X_train, np.ravel(y_train))
predictions = model_gbr.predict(X_test)
print("Root Mean squared error: %.2f" % sqrt(mean_squared_error(y_test, predictions))) # Root mean squared error

