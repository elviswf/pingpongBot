import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from filterpy.stats import plot_covariance_ellipse
from filterpy.kalman import KalmanFilter
from filterpy.stats import plot_covariance_ellipse
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
from physics import book_plots as bp


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
    zs = np.array([xdf['x'], xdf[ax]]).T
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
    plt.legend(loc=1)
    plt.ylabel(ax)
    plt.xlabel('t')
    plt.xlim((-0.1, 0.6))


def run_simulate(xdf, ax, R_std, Q_std):
    mu, zs1 = kf_simulate(xdf, R_std, Q_std, ax=ax)
    plot_bp(mu, zs1, ax)
    return mu


def plotdf(df, fig3d, color='r'):
    sc = fig3d.scatter(df['x'], df['y'], df['z'], c=color, alpha=0.3)
    fig3d.set_xlabel('x')
    fig3d.set_ylabel('y')
    fig3d.set_zlabel('z')
    plt.show()
    return sc


def kf_filter(xdf, R_std, Q_std, pici, save_dir='kf_pics'):
    fig = plt.figure(figsize=(12, 8))
    coordinates = ['y', 'z']
    ydf = pd.DataFrame(columns=['t', 'x', 'x_std', 'y', 'y_std', 'z', 'z_std'])
    ydf['t'] = xdf['t']
    for i, ax in enumerate(coordinates):
        plt.subplot(2, 2, i + 1)
        my = run_simulate(xdf, ax, R_std, Q_std)
        ydf[ax] = my[:, 2, 0]
        ydf[ax+'_std'] = my[:, 3, 0]
    # plt.subplot(2, 2, 3)
    # ax = 'z'
    # bounce_t = (xdf[xdf['z'] == xdf['z'].min()]['t']).values[0]
    # xdf_before = xdf[xdf['t'] < bounce_t]
    # mu, zs1 = kf_simulate(xdf_before, R_std, Q_std, ax=ax)
    # plot_bp(mu, zs1, ax)
    # bounce_t = xdf[xdf['z'] < 50]['t'].mean()
    # ydf.loc[ydf['t'] > bounce_t, 'z'] = xdf.loc[ydf['t'] > bounce_t, 'z']
    # ydf.loc[xdf['t'] < bounce_t, 'z'] = mu[:, 2, 0]
    # ydf.loc[xdf['t'] < bounce_t, 'z_std'] = mu[:, 3, 0]
    ydf['z'] = xdf['z']
    ax1 = fig.add_subplot(224, projection='3d')
    ax1.auto_scale_xyz([100, 1000], [-200, 3000], [0, 400])
    sc1 = plotdf(xdf, ax1, color='r')
    sc2 = plotdf(ydf, ax1, color='b')
    ax1.legend([sc1, sc2], ['data', 'filtered data'])
    plt.savefig('data/' + save_dir + '/kf_' + str(pici) + '.png')
    plt.close()
    return ydf


def run_kf(dfs, save_dir='kf_pics', R_std=0.35, Q_std=0.04):
    ydfs = []
    for i, xdf in enumerate(dfs):
        ydf = kf_filter(xdf, R_std, Q_std, i, save_dir)
        ydfs.append(ydf)
    return ydfs


# idxs = [0, 6, 9, 10, 20, 22, 26, 44, 49]
# idx = 40
idxs = [26]
for idx in idxs:
    xdf = dfs[idx]
    R_std = 0.36
    Q_std = 0.01
    ydf = kf_filter(xdf, R_std, Q_std, idx)


# dfs[75]
#
# ydfs = []
# for i, xdf in enumerate(dfs):
#     R_std = 0.35
#     Q_std = 0.04
#     ydf = kf_filter(xdf, R_std, Q_std, i)
#     ydfs.append(ydf)
#
# import pickle
# pickle.dump(ydfs, open('data/ydfs.p', 'wb'))
# ydfs = pickle.load(open('data/ydfs.p', 'rb'))
#
# ydfs[0].to_csv("trajectory_kf0.csv")
