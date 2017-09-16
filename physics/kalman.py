from filterpy.stats import plot_covariance_ellipse
from filterpy.kalman import KalmanFilter
import numpy as np
from filterpy.stats import plot_covariance_ellipse
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
from physics import book_plots as bp
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('data/data.csv')
ids = df[df.isnull().any(axis=1)].index
ids = [-1] + list(ids) + [len(df)]

dfs = []
for i in range(len(ids) - 1):
    dfs.append(df.iloc[ids[i] + 1:ids[i + 1]])


def tracker1(R_std, Q_std):
    tracker = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0  # time step

    tracker.F = np.array([[1, dt, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, dt],
                          [0, 0, 0, 1]])
    tracker.u = 0.
    tracker.H = np.array([[1, 0, 0, 0],
                          [0, 0, 1, 0]])

    tracker.R = np.eye(2) * R_std ** 2
    q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std ** 2)
    tracker.Q = block_diag(q, q)
    tracker.x = np.array([[0, 0, 0, 0]]).T
    tracker.P = np.eye(4) * 500.
    return tracker


# simulate robot movement
def kf_simulate(xdf, R_std, Q_std, ax='z'):
    zs = np.array([xdf['t'], xdf[ax]]).T
    zs = np.expand_dims(zs, axis=2)

    robot_tracker = tracker1(R_std, Q_std)
    mu, cov, _, _ = robot_tracker.batch_filter(zs)

    for x, P in zip(mu, cov):
        # covariance of x and y
        cov = np.array([[P[0, 0], P[2, 0]],
                        [P[0, 2], P[2, 2]]])
        mean = (x[0, 0], x[2, 0])
        # plot_covariance_ellipse(mean, cov=cov, fc='g', std=3, alpha=0.5)
    return mu, zs


# plot results
def plot_bp(mu, zs, ax):
    bp.plot_filter(mu[:, 0], mu[:, 2])
    bp.plot_measurements(zs[:, 0], zs[:, 1])
    plt.legend(loc=2)
    plt.ylabel(ax)
    plt.xlabel('t')
    plt.xlim((-0.1, 0.6))


def run_simulate(xdf, ax, R_std, Q_std):
    mu, zs1 = kf_simulate(xdf, R_std, Q_std, ax=ax)
    plot_bp(mu, zs1, ax)
    return mu


def plotdf(df, fig3d, color='r'):
    sc = fig3d.scatter(df['x'], df['y'], df['z'], c=color)
    fig3d.set_xlabel('x')
    fig3d.set_ylabel('y')
    fig3d.set_zlabel('z')
    plt.show()
    return sc


def kf_filter(xdf, R_std, Q_std):
    fig = plt.figure(figsize=(12, 8))
    coordinates = ['x', 'y', 'z']
    ydf = pd.DataFrame(columns=['t', 'x', 'x_std', 'y', 'y_std', 'z', 'z_std'])
    ydf['t'] = xdf['t']
    for i, ax in enumerate(coordinates):
        plt.subplot(2, 2, i + 1)
        my = run_simulate(xdf, ax, R_std, Q_std)
        ydf[ax] = my[:, 2, 0]
        ydf[ax+'_std'] = my[:, 3, 0]

    ax1 = fig.add_subplot(224, projection='3d')
    ax1.auto_scale_xyz([100, 1000], [-200, 3000], [0, 400])
    sc1 = plotdf(xdf, ax1, color='r')
    sc2 = plotdf(ydf, ax1, color='b')
    ax1.legend([sc1, sc2], ['data', 'filtered data'])
    return ydf


xdf = dfs[22]
R_std = 2
Q_std = 1
ydf = kf_filter(xdf, R_std, Q_std)
plt.savefig()


