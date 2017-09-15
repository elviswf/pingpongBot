import torch
import torch.nn as nn
from torch.autograd import Variable
from DataLoader import load_data, parse_data
import DataLoader as dl
import torch.optim as optim
import time
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class PPNet(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(PPNet, self).__init__()
        self.nIn = nIn
        self.nHidden = nHidden
        self.nOut = nOut

        self.net = nn.Sequential(nn.Linear(nIn, nHidden),
                                 nn.Tanh(),
                                 nn.Linear(nHidden, nHidden),
                                 nn.Tanh(),
                                 nn.Linear(nHidden, nOut),)
                                 #nn.ReLU())

    def forward(self, x):
        return self.net(x)


def train_step(model, criterion, optimizer, X, Y):

    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, Y)
    loss.backward()
    optimizer.step()

    return loss.data[0], pred


def train(model, criterion, optimizer, X,Y, epoch):
    model.train()
    end = time.time()

    N = X.size(0)
    X, Y = Variable(X), Variable(Y)
    batchsize = 64
    losses = 0
    for i in range(0,N,batchsize):
        stop = i + batchsize
        if stop>=N:
            stop = N-1
        input = X[i:stop,:]
        target = Y[i:stop,:]

        loss, pred = train_step(model, criterion, optimizer, input, target)
        losses += loss

    if epoch % 10 == 0:
        print('Epoch {0}, loss:{1}'.format(epoch, losses/math.ceil(N/batchsize)))


def validate(model, criterion, X, Y, epoch):
    model.eval()
    pred = model(Variable(X))
    loss = criterion(pred,Variable(Y))
    print('Validate Epoch {0}, loss {1}'.format(epoch, loss.data[0]))



def main():
    X_train, X_test, Y_train, Y_test = load_data()
    X_train, Y_train = torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float()
    X_test, Y_test = torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float()

    model = PPNet(16,40,3)
    criterion = nn.L1Loss()
    lr = 1e-3
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    num_epoch = 5000

    for epoch in range(num_epoch):
        '''
        if (epoch+1)%100==0:
            lr *= 0.2
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        '''
        train(model, criterion, optimizer, X_train, Y_train, epoch)
        if epoch % 5 == 0:
            validate(model, criterion, X_test, Y_test, epoch)

    return model


def vis(filepath):
    model = PPNet(16, 40, 3)
    state_dict = torch.load(filepath)
    model.load_state_dict(state_dict)
    X_train, X_test, Y_train, Y_test = load_data()
    val_X = torch.from_numpy(X_test).float()
    pred = model(Variable(val_X))
    pred = pred.data
    import matplotlib.pyplot as plt

    meanX = np.mean(np.abs(pred.numpy()[:, 0] - Y_test[:, 0]))
    meanY = np.mean(np.abs(pred.numpy()[:, 1] - Y_test[:, 1]))
    meanZ = np.mean(np.abs(pred.numpy()[:, 2] - Y_test[:, 2]))
    print('mean X:{0}, mean Y:{1}, mean Z:{2}'.format(meanX,meanY,meanZ))
    plt.plot(pred.numpy()[1:10,1],'*')
    plt.plot(Y_test[1:10,1],'s')
    plt.show()


def predict_and_plot(model, X, Y):
    X = torch.from_numpy(X).float()
    pred = model(Variable(X))
    pred = pred.data
    import matplotlib.pyplot as plt
    pred = pred.numpy()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(pred[:,0], pred[:,1], pred[:,2])
    #plt.scatter(pred[:,0], pred[:,1], pred[:2], 'o',label='pred')
    #plt.scatter(Y[:, 0],Y[:, 1],Y[:, 2],'^',label='ground truth')
    #plt.legend(loc=3)
    ax.scatter(Y[:, 0],Y[:, 1],Y[:, 2])
    plt.show()


def visline(filepath):
    model = PPNet(16, 40, 3)
    state_dict = torch.load(filepath)
    model.load_state_dict(state_dict)
    # generate test set
    data_list = parse_data()
    n_axis = 3

    for d, data in enumerate(data_list):
        X = []
        Y = []
        data[:, 1:] = data[:, 1:] / 1000.0
        key_points = dl.select_keypoint(data)
        if key_points is None:
            break
        key_points += [key_points[-1]]
        sample = np.zeros(16)
        s = 0
        for k in range(len(key_points) - 1):
            sample[s:s + n_axis] = data[key_points[k]][1:1 + n_axis]
            sample[s + n_axis] = data[key_points[k + 1]][0] - data[key_points[k]][0]

            s = s + n_axis + 1

        for i in range(key_points[-1] + 1, data.shape[0]):
            key_points[-1] = i
            sample_new = np.zeros(16)
            sample_new[:15] = sample[:15]
            sample_new[-1] = data[key_points[-1]][0] - data[key_points[-2]][0]
            X.append(sample_new)
            Y.append(data[i][1:1 + n_axis])
        X = np.vstack(X)
        Y = np.vstack(Y)
        predict_and_plot(model, X, Y)


if __name__=='__main__':
    testonly = 1
    if not testonly:
        model = main()
        torch.save(model.state_dict(),'model.pt')

        vis('model.pt')
    else:
        visline('model.pt')