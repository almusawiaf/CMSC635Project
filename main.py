# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 12:04:48 2023

@author: Ahmad Al Musawi
"""

from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

def preprocessing(df):
    print('preprocessing...')
    return df

def split_labels(df, cols):
    '''split the dataframe into predicting table and labels
       df: given dataset
       cols: list of labels
    '''
    return df[[i for i in df if i not in cols]], df[cols]
    

def SVM_Model(X, Y):
    ''' X is the given dataset
        Y is the labels list
        Split the dataset into training and testing sets
        return y_pred and y_test'''

    print('implementing SVM...')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = SVC(kernel='rbf', C=1.0) # Gaussian radial basis function (RBF) kernel
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred, y_test    

def CART_model(X,Y):
    ''' X is the given dataset
        Y is the labels list
        Split the dataset into training and testing sets'''

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred, y_test    

def PCA_model(X):
    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)
    df_pca = pd.DataFrame(data=X_pca)
    print(df_pca.shape)
    return df_pca

def CE_Model(X):
    embedding = SpectralEmbedding(n_components=2)
    X_CE = embedding.fit_transform(X)
    
    print(X_CE.shape)
    return X_CE


# Load the text file into a DataFrame
df1 = pd.read_csv('processed.cleveland.data', delimiter=',', header=None)
df2 = pd.read_excel('CTG.xls', sheet_name = 'Raw Data')


# Display the DataFrame
print(df1)
print(df2)



