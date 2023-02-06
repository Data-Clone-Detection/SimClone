import numpy as np
import json
import itertools
import pickle
import pandas as pd
import feature_utils
import random
import multiprocessing
import traceback
import copy
import sys
from time import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import torch.nn.functional as F
import torchvision.transforms as T
import torch
import torch.nn as nn
from torchvision.transforms.functional import InterpolationMode
import os
from scipy.linalg import block_diag
from datetime import datetime


def extract_feature(pair_order, index, pd_dict):
    if index % 10 == 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"index {index} : Time {current_time}")
    start_frac = 0
    threshold_frac = 10
    end_frac = 100
    try:
        data_object = {}
        data_object['img_index'] = index
        data_object['file_1_index'] = pair_order[0]
        data_object['file_2_index'] = pair_order[1]
        df_1 = copy.deepcopy(pd_dict.get(pair_order[0]))
        df_2 = copy.deepcopy(pd_dict.get(pair_order[1]))
        if df_1.shape[0] < 5 or df_2.shape[0] < 5:
            return None
        # self-pair, contain type 1 clone
        if data_object['file_1_index'] == data_object['file_2_index']:
            if random.choice([0, 1]) == 1:
                ratio = random.randint(threshold_frac, end_frac) / 100.0
                data_object['target'] = 1
            else:
                ratio = random.randint(start_frac, threshold_frac) / 100.0
                data_object['target'] = 0
            if random.choice([0, 1]) == 1:
                df_1_clone = df_1.sample(frac=ratio, random_state=index)
            else:
                df_1_clone = df_1.sample(frac=ratio, axis='columns', random_state=index)
            data_object['df_1'] = df_1
            data_object['df_2'] = df_1_clone
            data_object['clone_frag'] = df_1_clone

        else:
            if random.choice([0, 1]) == 1:  # contain type 3 duplicate
                data_object['target'] = 1
                dup_df, from_df, dup_case, sample = feature_utils.generate_duplicate_df(
                    df_1, df_2, random_seed=index, frac_start=threshold_frac, frac_end=end_frac)
            else:
                data_object['target'] = 0
                dup_df, from_df, dup_case, sample = feature_utils.generate_duplicate_df(
                    df_1, df_2, random_seed=index, frac_start=start_frac, frac_end=threshold_frac)
            data_object['df_1'] = from_df
            data_object['df_2'] = dup_df
            data_object['clone_frag'] = sample

        # suffle the pandas dataframe
        df_1 = data_object['df_1']
        df_2 = data_object['df_2']
        df_1 = df_1.sample(frac=1).reset_index(drop=True)
        df_1 = df_1.sample(frac=1, axis=1)
        df_2 = df_2.sample(frac=1).reset_index(drop=True)
        df_2 = df_2.sample(frac=1, axis=1)
        data_object['df_1'] = df_1
        data_object['df_2'] = df_2

        # calcuate LTC's feature similarity
        data_object['ltc_sim_features'] = feature_utils.df_csv_sim_features(
            copy.deepcopy(df_1), copy.deepcopy(df_2))
        # calculate similarity matrix (our method)
        sim_type, sim_mat_list = feature_utils.val_sim(
            copy.deepcopy(df_1), copy.deepcopy(df_2))

        # for machine learning methods
        top100_val_list = []
        for sim_mat in sim_mat_list:
            top100_val_list.append(feature_utils.get_top_k(sim_mat, 100))
        data_object['ml_sim_features'] = top100_val_list
        return data_object

    except:
        print(f"error in {index}")
        traceback.print_exc()
        return None



with open(f'UCI_pd_list_0924.pkl', 'rb') as f:
    pd_dict = pickle.load(f)

pair_order_list = list(itertools.combinations(pd_dict.keys(), 2))
for i in range(0, len(pd_dict.keys())):
    pair_order_list.append((str(i), str(i)))

data_objects = []
random.shuffle(pair_order_list)
# extract_feature(pair_order_list[46],46,pd_dict)

from itertools import repeat

with ProcessPoolExecutor(max_workers=46) as executor:
    for r in executor.map(extract_feature, pair_order_list, range(len(pair_order_list)), repeat(pd_dict)):
        if r:
            data_objects.append(r)
with open(f'UCI_data_0_10_100.pkl', 'wb') as f:
    pickle.dump(data_objects, f)