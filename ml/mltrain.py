import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('data/data.csv')
ids = df[df.isnull().any(axis=1)].index
ids = [-1] + list(ids) + [len(df)]

dfs = []
for i in range(len(ids) - 1):
    dfs.append(df.iloc[ids[i] + 1:ids[i + 1]])

df_label = pd.read_csv("data/label.txt", sep='\t', header=None)
df_label.head()

idxs = list(df_label[df_label[1] == 0].index)

id_train = idxs[:100]
id_test = idxs[100:]







