# -*- coding: utf-8 -*-
"""
@Time    : 9/12/2017 8:52 PM
@Author  : Elvis
"""
"""
 formula.py
  
"""
import pandas as pd

df = pd.read_csv('data/data.csv')
ids = df[df.isnull().any(axis=1)].index
ids = [-1] + list(ids) + [len(df)]

dfs = []
for i in range(len(ids) - 1):
    dfs.append(df.iloc[ids[i] + 1:ids[i + 1]])

len(dfs)
dfs[-1]
xdf = dfs[-1]

xdf

from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plotdf(df, fig3d):
    fig3d.scatter(df['x'], df['y'], df['z'])
    fig3d.set_xlabel('x')
    fig3d.set_ylabel('y')
    fig3d.set_zlabel('z')
    plt.show()


fig3d = plt.figure().gca(projection='3d')
for i in range(10):
    plotdf(dfs[i], fig3d)












