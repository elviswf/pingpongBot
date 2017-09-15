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
# plt.axis('equal')
# fig3d.set_xlim([100, 1000])
# fig3d.set_xlim([-200, 3000])
# fig3d.set_xlim([0, 400])

fig3d.auto_scale_xyz([100, 1000], [-200, 3000], [0, 400])
for i in range(10):
    plotdf(dfs[i], fig3d)

from pykalman import KalmanFilter
import numpy as np

kf = KalmanFilter(transition_matrices=[[1, 0, 1, 0], [0, 1, 0, 1]], observation_matrices=[[0.1, 0.5], [-0.3, 0.0]])
measurements = np.asarray([[1, 0], [0, 0], [0, 1]])  # 3 observations
kf = kf.em(measurements, n_iter=5)
(filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)
