# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 12:04:48 2023

@author: Ahmad Al Musawi
"""

from sklearn.manifold import SpectralEmbedding
import pandas as pd

# Load the text file into a DataFrame
df = pd.read_csv('processed.cleveland.data', delimiter=',', header=None)

# Display the DataFrame
print(df)

embedding = SpectralEmbedding(n_components=2)
df1 = embedding.fit_transform(df)
print(df1.shape)
