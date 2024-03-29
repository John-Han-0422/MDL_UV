# -*- coding: utf-8 -*-

import pandas as pd
from mulutils import build_iterator, train, configmul
from models import mulNet
from sklearn.model_selection import train_test_split
import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

def random_strati_smp(samples, num):
    num = int(num/10)
    # Divide the training data into 10 equivalent intervals by y
    samples['y_interval'] = pd.cut(samples['gcr'], bins=10)

    # Randomly sample 10% of data from each y interval
    stra_sample = pd.DataFrame()
    for interval in samples['y_interval'].unique():
        interval_data = samples[samples['y_interval'] == interval]
        interval_sample = interval_data.sample(n = num, random_state=None)
        stra_sample = pd.concat([stra_sample, interval_sample])

    # Remove the y_interval column from the training set
    stra_sample = stra_sample.drop(columns=['y_interval'])
    # stra_sample = stra_sample.sample(frac=1, random_state=None)
    stra_sample = shuffle(stra_sample, random_state=None, n_samples=int(len(stra_sample)))
    # split the dataframe into training and testing sets
    train_set, test_set = train_test_split(stra_sample, test_size=0.3, random_state=None)
    # Print the shapes of the resulting datasets
    print(f"Original dataset shape: {samples.shape}")
    print(f"Training set shape: {train_set.shape}")
    print(f"Testing set shape: {test_set.shape}")

    y_train = train_set['gcr']
    y_test = test_set['gcr']
    X_train = train_set[desired_bands]
    X_test = test_set[desired_bands]

    return y_train,y_test,X_train,X_test

desired_bands = ['F1_blue', 'F1_green', 'F1_red', 'F1_nir', 'F1_swir1', 'F1_swir2',
                 'F2_NDVI','F2_EVI',
                 'F3_r_con', 'F3_r_ent', 'F3_r_var', 'F3_r_ASM', 'F3_r_IDM', 'F3_r_cor',
                 'F3_g_con', 'F3_g_ent', 'F3_g_var', 'F3_g_ASM', 'F3_g_IDM', 'F3_g_cor',
                 'F3_b_con', 'F3_b_ent', 'F3_b_var', 'F3_b_ASM', 'F3_b_IDM', 'F3_b_cor',
                 'F3_ndvi_con', 'F3_ndvi_ent', 'F3_ndvi_var', 'F3_ndvi_ASM', 'F3_ndvi_IDM', 'F3_ndvi_cor',
                 'F4_elev', 'F4_slope','F5_time_labels']

columns_to_normalize = ['F3_r_con', 'F3_r_ent', 'F3_r_var', 'F3_r_ASM', 'F3_r_IDM', 'F3_r_cor',
                 'F3_g_con', 'F3_g_ent', 'F3_g_var', 'F3_g_ASM', 'F3_g_IDM', 'F3_g_cor',
                 'F3_b_con', 'F3_b_ent', 'F3_b_var', 'F3_b_ASM', 'F3_b_IDM', 'F3_b_cor',
                 'F3_ndvi_con', 'F3_ndvi_ent', 'F3_ndvi_var', 'F3_ndvi_ASM', 'F3_ndvi_IDM', 'F3_ndvi_cor',
                 'F4_elev', 'F4_slope']

is_sparse = True
data = pd.read_csv("samples.csv")
scaler = StandardScaler()
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
c = sorted(desired_bands)
x = data[desired_bands]
y = data['gcr']
pre = 0
group_index = []
i = 1
while i < len(c):
    a=c[i][:2]
    b =  c[pre][:2]
    while i < len(c) and c[i][:2] == c[pre][:2]:
        i += 1
    group_index.append((pre, i))
    pre = i

y_train, y_test, x_train, x_test = random_strati_smp(data, 250000)
config = configmul()
train_iter = build_iterator(pd.concat([x_train,y_train],axis=1), config)
dev_iter = build_iterator(pd.concat([x_test,y_test], axis=1), config)
if not is_sparse:
    group_index.append(())
model = mulNet(config, x, sparse_group_index = group_index[-1], group_index=group_index[:-1])
model.to(config.device)
train(config, model, train_iter, dev_iter)
