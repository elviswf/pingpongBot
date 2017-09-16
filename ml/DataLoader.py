import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def parse_data():
    df = pd.read_excel('data.xlsx')
    del df['Unnamed: 4']

    data_list = []
    start_idx = 0
    for i in (np.where(np.isnan(df.time))[0]):
        data_list.append(np.array(df.iloc[start_idx:i, 0:4]))
        start_idx = i + 1

    data_list.append(np.array(df.iloc[start_idx:, 0:4]))
    return data_list


def find_bouncepoint(data):
    """ Find the bounce point of each trajectory."""
    z = data[:,-1]
    idx = np.argmin(z)
    return idx


def select_keypoint(data):
    """ Select key points from a trajectory."""
    # phase I
    bouncepoint = find_bouncepoint(data)
    endofphase1 = 0
    """
            for i in range(bouncepoint, 2, -1):
            if data[i-1,-1] > data[i,-1]:
                continue
            else:
                endofphase1 = i
                break
        if endofphase1 > 3:
            keypoint = random.sample(list(range(endofphase1)), 3)
            keypoint += [endofphase1]
            return sorted(keypoint)
        else:
            return [2, 4, 6, 8]
    """

    return [2,4,6,8]


def make_pos_data():
    data_list = parse_data()
    label = np.loadtxt('label.txt')
    label.astype(np.int32)
    idx_list = list(np.where(label[:,1] == 0)[0])
    n_axis = 3

    X = []
    Y = []

    for d, data in enumerate(data_list):
        if d not in idx_list:
            continue
        data[:,1:] = data[:,1:] / 1000.0
        key_points = select_keypoint(data)
        if key_points is None:
            break
        key_points += [key_points[-1]]
        sample = np.zeros(16)
        s = 0
        for k in range(len(key_points) - 1):
            sample[s:s + n_axis] = data[key_points[k]][1:1 + n_axis]
            sample[s + n_axis] = data[key_points[k + 1]][0] - data[key_points[k]][0]

            s = s + n_axis + 1

        for i in range(key_points[-1]+1, data.shape[0]):
            key_points[-1] = i
            sample_new = np.zeros(16)
            sample_new[:15] = sample[:15]
            sample_new[-1] = data[key_points[-1]][0] - data[key_points[-2]][0]
            X.append(sample_new)
            Y.append(data[i][1:1 + n_axis])
    X = np.vstack(X)
    X[:, [3, 7, 11, 15]] = X[:, [3, 7, 11, 15]] * 20
    Y = np.vstack(Y)
    return X, Y


def make_XY_data():
    data_list = parse_data()
    n_dim = 2
    s_dim = 18
    train, test = train_test_split(range(200), random_state=12)
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for d,data in enumerate(data_list):
        tmpX = []
        tmpY = []
        data[:, 1:] = data[:, 1:] / 1000.0
        kp = select_keypoint(data)
        kp += [kp[-1]]
        sample = np.zeros(s_dim)
        s = 0
        for k in range(len(kp) - 1):
            sample[s:s+n_dim] = data[kp[k]][1:1+n_dim]
            sample[s+n_dim] = data[kp[k+1]][0] - data[kp[k]][0]
            s = s+n_dim+1
            if k == len(kp)-2:
                break
            sample[s:s+n_dim] = data[kp[k+1]][1:1+n_dim] - data[kp[k]][1:1+n_dim]
            s = s+n_dim

        for i in range(kp[-1]+1, data.shape[0]):
            kp[-1] = i
            sample_new = np.zeros(s_dim)
            sample_new[:s_dim-1] = sample[:s_dim-1]
            sample_new[-1] = data[kp[-1]][0] - data[kp[-2]][0]
            tmpX.append(sample_new)
            tmpY.append(np.hstack((data[i][1:1 + n_dim], data[kp[-1]][1:1+n_dim]-\
                               data[kp[-2]][1:1+n_dim])))

        tmpX, tmpY = np.vstack(tmpX), np.vstack(tmpY)
        if d in train:
            X_train.append(tmpX)
            Y_train.append(tmpY)
        else:
            X_test.append(tmpX)
            Y_test.append(tmpY)

    return X_train, Y_train, X_test, Y_test


def make_XZ_data():
    data_list = parse_data()
    n_dim = 2
    s_dim = 18
    train, test = train_test_split(range(200), random_state=12)
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for d, data in enumerate(data_list):
        tmpX = []
        tmpY = []
        data[:, 1:] = data[:, 1:] / 1000.0
        kp = select_keypoint(data)
        kp += [kp[-1]]
        sample = np.zeros(s_dim)
        s = 0
        for k in range(len(kp) - 1):
            sample[s:s + n_dim] = data[kp[k]][[1,3]]
            sample[s + n_dim] = data[kp[k + 1]][0] - data[kp[k]][0]
            s = s + n_dim + 1
            if k == len(kp) - 2:
                break
            sample[s:s + n_dim] = data[kp[k + 1]][[1,3]] - data[kp[k]][[1,3]]
            s = s + n_dim

        for i in range(kp[-1] + 1, data.shape[0]):
            kp[-1] = i
            sample_new = np.zeros(s_dim)
            sample_new[:s_dim - 1] = sample[:s_dim - 1]
            sample_new[-1] = data[kp[-1]][0] - data[kp[-2]][0]
            tmpX.append(sample_new)
            tmpY.append(np.hstack((data[i][[1,3]], data[kp[-1]][[1,3]] - \
                                   data[kp[-2]][[1,3]])))

        tmpX, tmpY = np.vstack(tmpX), np.vstack(tmpY)
        if d in train:
            X_train.append(tmpX)
            Y_train.append(tmpY)
        else:
            X_test.append(tmpX)
            Y_test.append(tmpY)
    return X_train, Y_train, X_test, Y_test

def make_YZ_data():
    data_list = parse_data()
    n_dim = 2
    s_dim = 18
    train, test = train_test_split(range(200), random_state=12)
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for d, data in enumerate(data_list):
        tmpX = []
        tmpY = []
        data[:, 1:] = data[:, 1:] / 1000.0
        kp = select_keypoint(data)
        kp += [kp[-1]]
        sample = np.zeros(s_dim)
        s = 0
        for k in range(len(kp) - 1):
            sample[s:s + n_dim] = data[kp[k]][[2, 3]]
            sample[s + n_dim] = data[kp[k + 1]][0] - data[kp[k]][0]
            s = s + n_dim + 1
            if k == len(kp) - 2:
                break
            sample[s:s + n_dim] = data[kp[k + 1]][[2, 3]] - data[kp[k]][[2, 3]]
            s = s + n_dim

        for i in range(kp[-1] + 1, data.shape[0]):
            kp[-1] = i
            sample_new = np.zeros(s_dim)
            sample_new[:s_dim - 1] = sample[:s_dim - 1]
            sample_new[-1] = data[kp[-1]][0] - data[kp[-2]][0]
            tmpX.append(sample_new)
            tmpY.append(np.hstack((data[i][[2, 3]], data[kp[-1]][[2, 3]] - \
                                   data[kp[-2]][[2, 3]])))

        tmpX, tmpY = np.vstack(tmpX), np.vstack(tmpY)
        if d in train:
            X_train.append(tmpX)
            Y_train.append(tmpY)
        else:
            X_test.append(tmpX)
            Y_test.append(tmpY)
    return X_train, Y_train, X_test, Y_test


def make_posvec_data():
    data_list = parse_data()
    n_dim = 3
    s_dim = 25
    train, test = train_test_split(range(200), random_state=12)
    X_train, Y_train= [], []
    X_test, Y_test  = [], []

    for d,data in enumerate(data_list):
        tmpX = []
        tmpY = []
        data[:, 1:] = data[:, 1:] / 1000.0
        kp = select_keypoint(data)
        kp += [kp[-1]]
        sample = np.zeros(s_dim)
        s = 0
        for k in range(len(kp) - 1):
            sample[s:s+n_dim] = data[kp[k]][1:1+n_dim]
            sample[s+n_dim] = data[kp[k+1]][0] - data[kp[k]][0]
            s = s+n_dim+1
            if k == len(kp)-2:
                break
            sample[s:s+n_dim] = data[kp[k+1]][1:1+n_dim] - data[kp[k]][1:1+n_dim]
            s = s+n_dim

        for i in range(kp[-1]+1, data.shape[0]):
            kp[-1] = i
            sample_new = np.zeros(s_dim)
            sample_new[:s_dim-1] = sample[:s_dim-1]
            sample_new[-1] = data[kp[-1]][0] - data[kp[-2]][0]
            tmpX.append(sample_new)
            tmpY.append(np.hstack((data[i][1:1 + n_dim], data[kp[-1]][1:1+n_dim]-\
                               data[kp[-2]][1:1+n_dim])))
            #if d in train:
            #    X_train.append(sample_new)
            #    Y_train.append(np.hstack((data[i][1:1 + n_dim], data[kp[-1]][1:1+n_dim]-\
            #                   data[kp[-2]][1:1+n_dim])))
            #else:
            #    X_test.append(sample_new)
            #    Y_test.append(np.hstack((data[i][1:1 + n_dim], data[kp[-1]][1:1 + n_dim] - \
            #                             data[kp[-2]][1:1 + n_dim])))
        tmpX, tmpY = np.vstack(tmpX), np.vstack(tmpY)
        if d in train:
            X_train.append(tmpX)
            Y_train.append(tmpY)
        else:
            X_test.append(tmpX)
            Y_test.append(tmpY)
    return X_train, Y_train, X_test, Y_test


def load_data():
    """
    X, Y = make_posvec_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=12)
    return X_train, X_test, Y_train, Y_test
    :return: 
    """

    X_train, Y_train, X_test, Y_test = make_posvec_data()
    X_train = np.vstack(X_train)
    X_train[:, [3, 10, 17, 24]] = X_train[:, [3, 10, 17, 24]] * 20
    Y_train = np.vstack(Y_train)
    X_test = np.vstack(X_test)
    X_test[:, [3, 10, 17, 24]] = X_test[:, [3, 10, 17, 24]] * 20
    Y_test = np.vstack(Y_test)

    scaler1 = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler1.transform(X_train)
    X_test = scaler1.transform(X_test)
    scaler2 = preprocessing.StandardScaler().fit(Y_train)
    Y_train = scaler2.transform(Y_train)
    Y_test = scaler2.transform(Y_test)
    return X_train, X_test, Y_train, Y_test


def load_XY_data():
    X_train, Y_train, X_test, Y_test = make_XY_data()
    X_train = np.vstack(X_train)
    X_train[:, [2, 7, 12, 17]] = X_train[:, [2, 7, 12, 17]] * 20

    Y_train = np.vstack(Y_train)
    X_test = np.vstack(X_test)
    X_test[:, [2, 7, 12, 17]] = X_test[:, [2, 7, 12, 17]] * 20
    Y_test = np.vstack(Y_test)

    scaler1 = preprocessing.MinMaxScaler()
    X_train = scaler1.fit_transform(X_train)
    X_test = scaler1.transform(X_test)
    scaler2 = preprocessing.MinMaxScaler()
    Y_train = scaler2.fit_transform(Y_train)
    Y_test = scaler2.transform(Y_test)
    #Y_train, Y_test = scaler1.transform(Y_train)
    return X_train, X_test, Y_train, Y_test


def get_scalers(X_train,Y_train):
    X_train = np.vstack(X_train)

    Y_train = np.vstack(Y_train)

    scaler1 = preprocessing.MinMaxScaler().fit(X_train)

    scaler2 = preprocessing.MinMaxScaler().fit(Y_train)
    return scaler1, scaler2

def load_XZ_data():
    X_train, Y_train, X_test, Y_test = make_XZ_data()
    X_train = np.vstack(X_train)
    X_train[:, [2, 7, 12, 17]] = X_train[:, [2, 7, 12, 17]] * 20
    Y_train = np.vstack(Y_train)
    X_test = np.vstack(X_test)
    X_test[:, [2, 7, 12, 17]] = X_test[:, [2, 7, 12, 17]] * 20
    Y_test = np.vstack(Y_test)
    return X_train, X_test, Y_train, Y_test


def load_YZ_data():
    X_train, Y_train, X_test, Y_test = make_YZ_data()
    X_train = np.vstack(X_train)
    X_train[:, [2, 7, 12, 17]] = X_train[:, [2, 7, 12, 17]] * 20
    Y_train = np.vstack(Y_train)
    X_test = np.vstack(X_test)
    X_test[:, [2, 7, 12, 17]] = X_test[:, [2, 7, 12, 17]] * 20
    Y_test = np.vstack(Y_test)
    return X_train, X_test, Y_train, Y_test


def fit():
    import matplotlib.pyplot as plt
    data_list = parse_data()
    data = data_list[0]
    z1 = np.polyfit(data[:,0], data[:,3], 3)
    yvals = np.polyval(z1, data[:,0])
    plt.plot(data[:,0],data[:,3],'*')
    plt.plot(data[:,0], yvals,'r')
    plt.show()


def test():
    '''
    data_list = parse_data()
    find_bouncepoint(data_list[1])
    import matplotlib.pyplot as plt
    for i in range(20):
        plt.plot(data_list[i][:,-1])
    plt.show()
    print select_keypoint(data_list[2])
    '''
    X,Y = make_posvec_data()

    print X[:10,]
    print Y.shape


def visialize3D():
    data_list = parse_data()
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    for d, data in enumerate(data_list):
        fig = plt.figure()
        ax = fig.add_subplot(221,projection='3d')
        ax.scatter(data[:, 1], data[:, 2], data[:, 3])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax = fig.add_subplot(222)
        plt.plot(data[:,1],data[:,2],'*')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax = fig.add_subplot(223)
        plt.plot(data[:, 1], data[:, 3], '*')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax = fig.add_subplot(224)
        plt.plot(data[:, 2], data[:, 3], '*')
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        #plt.show()
        #break
        plt.savefig('./traj/trajectory-'+str(d)+'.png')

#visialize3D()
#make_posvec_data()