import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd


df = pd.read_csv('data/data.csv')
ids = df[df.isnull().any(axis=1)].index
ids = [-1] + list(ids) + [len(df)]

dfs = []
for i in range(len(ids) - 1):
    dfs.append(df.iloc[ids[i] + 1:ids[i + 1]])


