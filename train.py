import pickle
import itertools
from tqdm import tqdm
import copy
import argparse
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from multiprocessing import Process, Manager, Pool
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from statistics import mean

with open(f'data/UCI_data_0_10_100.pkl', 'rb') as f:
    xs = pickle.load(f)

datas = []
mlp_datas = []
k = 10
for x in xs:
    data = []
    mlp_data = []
    data.extend(x['ltc_sim_features'][0:3])
    for sim_mat in x['ml_sim_features']:
        top_k = sim_mat[:k]
        if len(top_k) == 0:
            data.append(0.)
        else:
            data.append(mean(top_k))
        mlp_data.extend(top_k)
        mlp_data.extend([0] * (k - len(top_k)))
    mlp_data.append(x['target'])
    mlp_datas.append(mlp_data)
    data.append(x['target'])
    datas.append(data)

df = pd.DataFrame(datas, columns=['col_header', 'row_header', 'cell_type', 'jaccard_str_row', 'jaccard_str_col',
                                  'jaccard_num_row', 'jaccard_num_col', 'mean_col', 'dev_col', 'mean_row', 'dev_row',
                                  'simhash_col', 'simhash_row', 'lev_col', 'lev_row', 'textrank_col', 'textrank_row',
                                  'target'])
print(df.target.value_counts())
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.1, random_state=42)
print('baseline LTC')
predictor_ltc = TabularPredictor(label='target', verbosity=0).fit(
    train_data=train[['col_header', 'row_header', 'cell_type', 'target']])

print(predictor_ltc.evaluate(test[['col_header', 'row_header', 'cell_type', 'target']]))

print('SimClone')
predictor = TabularPredictor(label='target',verbosity=0,path='ML_predictor').fit(train_data=train[
    ['jaccard_str_row', 'jaccard_str_col', 'jaccard_num_row', 'jaccard_num_col', 'mean_col', 'dev_col', 'mean_row',
     'dev_row', 'simhash_col', 'simhash_row', 'lev_col', 'lev_row', 'textrank_col', 'textrank_row', 'target']])

print(predictor.evaluate(test[['jaccard_str_row', 'jaccard_str_col', 'jaccard_num_row', 'jaccard_num_col', 'mean_col',
                               'dev_col', 'mean_row', 'dev_row', 'simhash_col', 'simhash_row', 'lev_col', 'lev_row',
                               'textrank_col', 'textrank_row', 'target']]))


