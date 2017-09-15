import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 8)

# intial parameters
n_iter = 50
sz = (n_iter,)  # size of array
x = -0.37727  # truth value (typo in example at top of p. 13 calls this z)
z = np.random.normal(x, 0.1, size=sz)  # observations (normal about x, sigma=0.1)

Q = 1e-5  # process variance

# allocate space for arrays
xhat = np.zeros(sz)  # a posteri estimate of x
P = np.zeros(sz)  # a posteri error estimate
xhatminus = np.zeros(sz)  # a priori estimate of x
Pminus = np.zeros(sz)  # a priori error estimate
K = np.zeros(sz)  # gain or blending factor

R = 0.1 ** 2  # estimate of measurement variance, change to see effect

# intial guesses
xhat[0] = 0.0
P[0] = 1.0

for k in range(1, n_iter):
    # time update
    xhatminus[k] = xhat[k - 1]
    Pminus[k] = P[k - 1] + Q

    # measurement update
    K[k] = Pminus[k] / (Pminus[k] + R)
    xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])
    P[k] = (1 - K[k]) * Pminus[k]

plt.figure()
plt.plot(z, 'k+', label='noisy measurements')
plt.plot(xhat, 'b-', label='a posteri estimate')
plt.axhline(x, color='g', label='truth value')
plt.legend()
plt.title('Estimate vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('Voltage')

# plt.figure()
# valid_iter = range(1, n_iter)  # Pminus not valid at step 0
# plt.plot(valid_iter, Pminus[valid_iter], label='a priori error estimate')
# plt.title('Estimated $\it{\mathbf{a \ priori}}$ error vs. iteration step', fontweight='bold')
# plt.xlabel('Iteration')
# plt.ylabel('$(Voltage)^2$')
# plt.setp(plt.gca(), 'ylim', [0, .01])
# plt.show()

import pandas as pd

df = pd.read_csv('data/data.csv')
ids = df[df.isnull().any(axis=1)].index
ids = [-1] + list(ids) + [len(df)]

dfs = []
for i in range(len(ids) - 1):
    dfs.append(df.iloc[ids[i] + 1:ids[i + 1]])

xdf = dfs[3]

from numpy.random import randn
import copy


class PosSensor1(object):
    def __init__(self, pos=(0, 0), vel=(0, 0), noise_std=1.):
        self.vel = vel
        self.noise_std = noise_std
        self.pos = [pos[0], pos[1]]

    def read(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]

        return [self.pos[0] + randn() * self.noise_std,
                self.pos[1] + randn() * self.noise_std]


from filterpy.stats import plot_covariance_ellipse
from filterpy.kalman import KalmanFilter
import numpy as np
from filterpy.stats import plot_covariance_ellipse
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
from physics import book_plots as bp


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
def kf_simulate(zs, R_std, Q_std, ax='z'):
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
def plot(mu, zs, ax):
    bp.plot_filter(mu[:, 0], mu[:, 2])
    bp.plot_measurements(zs[:, 0], zs[:, 1])
    plt.legend(loc=2)
    plt.ylabel(ax)
    plt.xlabel('t')
    plt.xlim((-0.1, 0.6))


def run(zs, ax):
    R_std = 1
    Q_std = 1
    mu, zs1 = kf_simulate(zs, R_std, Q_std, ax=ax)
    plot(mu, zs1, ax)


def plotdf(df, fig3d):
    fig3d.scatter(df['x'], df['y'], df['z'])
    fig3d.set_xlabel('x')
    fig3d.set_ylabel('y')
    fig3d.set_zlabel('z')
    plt.show()


xdf = dfs[49]
fig = plt.figure()
ax1 = fig.add_subplot(221, projection='3d')
ax1.auto_scale_xyz([100, 1000], [-200, 3000], [0, 400])
plotdf(xdf, ax1)
for i, ax in enumerate(['x', 'y', 'z']):
    plt.subplot(2, 2, i+2)
    run(xdf, ax)