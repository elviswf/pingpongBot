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
dfs[-1].to_csv("trace1.csv", index=False)
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
for i in range(50, 100):
    plotdf(dfs[i], fig3d)


def update(prior, measurement):
    x, P = prior  # mean and variance of prior
    z, R = measurement  # mean and variance of measurement

    y = z - x  # residual
    K = P / (P + R)  # Kalman gain

    x = x + K * y  # posterior
    P = (1 - K) * P  # posterior variance
    return x, P


def predict(posterior, movement):
    x, P = posterior  # mean and variance of posterior
    dx, Q = movement  # mean and variance of movement
    x = x + dx
    P = P + Q
    return x, P


import filterpy.kalman as kf

# x = predict(x, vx)
# x = update(x, mx)
kf.predict(x=10, P=3., u=1, Q=4)
x, P = kf.predict(x=10, P=3., u=1, Q=2**2)
x, P = kf.update(x=x, P=P, z=12, R=3.5**2)
x, P


from filterpy.kalman import KalmanFilter
import numpy as np
from filterpy.stats import plot_covariance_ellipse
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag

R_std = 0.35
Q_std = 0.04


def tracker1():
    tracker = KalmanFilter(dim_x=6, dim_z=3)
    dt = 1.0  # time step

    tracker.F = np.array([[1, dt, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 1, dt, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, dt],
                          [0, 0, 0, 0, 0, 1]])
    tracker.u = 0.
    tracker.H = np.array([[1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0]])

    tracker.R = np.eye(3) * R_std ** 2
    q = Q_discrete_white_noise(dim=3, dt=dt, var=Q_std ** 2)
    tracker.Q = block_diag(q, q, q)
    tracker.x = np.array([[0, 0, 0, 0, 0, 0]]).T
    tracker.P = np.eye(6) * 400.
    return tracker


plt.figure()
# simulate robot movement
zs = xdf
N = 30

# run filter
robot_tracker = tracker1()
mu, cov, _, _ = robot_tracker.batch_filter(zs)

for x, P in zip(mu, cov):
    # covariance of x and y
    cov = np.array([[P[0, 0], P[2, 0]],
                    [P[0, 2], P[2, 2]]])
    mean = (x[0, 0], x[2, 0])
    plot_covariance_ellipse(mean, cov=cov, fc='g', std=3, alpha=0.5)

# plot results
zs *= .3048  # convert to meters
bp.plot_filter(mu[:, 0], mu[:, 2])
bp.plot_measurements(zs[:, 0], zs[:, 1])
plt.legend(loc=2)
plt.xlim((0, 20))


