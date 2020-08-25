#! usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import gc

def data_read(s, r):
    Label = pd.read_csv('./AWE/Label.csv')

    vibr_list = ['Normal', 'Fault-1', 'Fault-2', 'Fault-3', 'Fault-4', 'Fault-5']
    vibr_all_normal, vibr_all_fault1, vibr_all_fault2, vibr_all_fault3, vibr_all_fault4, vibr_all_fault5 = [], [], [], [], [], []

    for i in range(1, 7):
        vibr = pd.read_csv('./AWE/{}.csv'.format(vibr_list[i-1]), nrows=r).drop('Index', axis=1).reset_index(drop=True).values.reshape([-1, 4000])
        vibr1 = np.array(vibr)

        if i == 1:
            for j in range(1, r+1):
                vibr_all_normal.append(vibr1[j-1])
        elif i == 2:
            for j in range(1, r+1):
                vibr_all_fault1.append(vibr1[j-1])
        elif i == 3:
            for j in range(1, r+1):
                vibr_all_fault2.append(vibr1[j-1])
        elif i == 4:
            for j in range(1, r+1):
                vibr_all_fault3.append(vibr1[j-1])
        elif i == 5:
            for j in range(1, r+1):
                vibr_all_fault4.append(vibr1[j-1])
        else:
            for j in range(1, r+1):
                vibr_all_fault5.append(vibr1[j-1])

        del vibr, vibr1
        gc.collect()

    vibr_all_normal = pd.DataFrame(vibr_all_normal).reset_index(drop=True).values.reshape((-1, 1))
    vibr_all_fault1 = pd.DataFrame(vibr_all_fault1).reset_index(drop=True).values.reshape((-1, 1))
    vibr_all_fault2 = pd.DataFrame(vibr_all_fault2).reset_index(drop=True).values.reshape((-1, 1))
    vibr_all_fault3 = pd.DataFrame(vibr_all_fault3).reset_index(drop=True).values.reshape((-1, 1))
    vibr_all_fault4 = pd.DataFrame(vibr_all_fault4).reset_index(drop=True).values.reshape((-1, 1))
    vibr_all_fault5 = pd.DataFrame(vibr_all_fault5).reset_index(drop=True).values.reshape((-1, 1))

    X_dataset = pd.concat([pd.DataFrame(vibr_all_normal),\
                              pd.DataFrame(vibr_all_fault1),\
                              pd.DataFrame(vibr_all_fault2),\
                              pd.DataFrame(vibr_all_fault3),\
                              pd.DataFrame(vibr_all_fault4),\
                              pd.DataFrame(vibr_all_fault5)], axis=0).reset_index(drop=True).values.reshape([-1, 1])

    num_samples = int((len(vibr_all_normal)/s))*6
    X_data = pd.concat([pd.DataFrame(X_dataset[0:num_samples * s])], axis=1).reset_index(drop=True).values.reshape((-1, s, 1))

    y_data = pd.concat([Label["L0"].loc[0:num_samples/6-1],\
                         Label["L1"].loc[0:num_samples/6-1],\
                         Label["L2"].loc[0:num_samples/6-1],\
                         Label["L3"].loc[0:num_samples/6-1],\
                         Label["L4"].loc[0:num_samples/6-1],\
                         Label["L5"].loc[0:num_samples/6-1]], axis=0).reset_index(drop=True).values.reshape((-1, 1))

    return X_data, y_data

def data_embedding(x, seq_len):
    X_data = []
    for i in range(0, len(x)):
        x_data = x[i]
        x_data = pd.DataFrame(x_data).values.reshape((8, int(seq_len / 8)))
        X_data.append(x_data)

    return X_data