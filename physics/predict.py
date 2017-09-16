import pandas as pd
from physics import kalman


def read_input(inputname='data/data_test.csv'):
    df = pd.read_csv(inputname)
    ids = df[df.isnull().any(axis=1)].index
    ids = [-1] + list(ids) + [len(df)]

    dfs = []
    for i in range(len(ids) - 1):
        dfs.append(df.iloc[ids[i] + 1:ids[i + 1]])
    return dfs


dfs = read_input(inputname='data/data_test.csv')
len(dfs)

ydfs = kalman.run_kf(dfs, save_dir='test_pics', R_std=0.35, Q_std=0.04)



