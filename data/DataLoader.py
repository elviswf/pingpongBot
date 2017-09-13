import pandas as pd
import numpy as np


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

#
#data_list = parse_data()
#print len(data_list)