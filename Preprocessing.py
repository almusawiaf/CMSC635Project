# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 11:21:34 2023

@author: Ashwini
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


#printing the first few rows
df = pd.read_csv("C:/Users/Yoges/Downloads/archive (2)/fetal_health.csv")

#Check null values
df.isnull().sum()

#check missing values
miss_values = df.columns[df.isnull().any()]
print(f"Missing values:\n{df[miss_values].isnull().sum()}")

#check correlated features
correlated_features = set()
correlation_matrix = df.corr()
for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
            
print (len(correlated_features))

#checking imbalance data
print('Checking how imbalance is the dataset:')
print(df['fetal_health'].value_counts())

#check duplicate records
print(df.duplicated().sum())
print('Labels counts duplicate: ')
print(df.loc[df.duplicated(), 'fetal_health'].value_counts())

# Removing duplicates
df.drop_duplicates(inplace=True)

# after removing duplicates again check how imbalance the dataset is:
print('Checking how imbalance is the dataset after removing duplicates:')
print(df['fetal_health'].value_counts())



