import pandas as pd
from .visualization import show_features
from .preprocess import FilterWrapper
from .filters import *
import numpy as np

def eval_kmean(pred, correct):
    if len(pred) != len(correct):
        raise ValueError("prediction and correct have different length")
    size = len(pred)
    d = {}
    for i in range(size):
        correct_arr = correct[i]
        correct_class = correct_arr.index(1)
        pred_class = pred[i]
        if correct_class not in d:
            d[correct_class] = {}
        if pred_class not in d[correct_class]:
            d[correct_class][pred_class] = 0
        d[correct_class][pred_class] = d[correct_class][pred_class] + 1
    g_correct_cnt = 0
    g_total_cnt = 0
    for c in d:
        pred_all = d[c]
        max_cnt = 0
        total_cnt = 0
        for pred_key in pred_all:
            pred_cnt = pred_all[pred_key]
            max_cnt = max(max_cnt, pred_cnt)
            total_cnt = total_cnt + pred_cnt
        g_correct_cnt = g_correct_cnt + max_cnt
        g_total_cnt = g_total_cnt + total_cnt
    accuracy = g_correct_cnt / g_total_cnt
    return accuracy

def read_kmeans_data(filename_labeled, filename_unlabeled, label_name):
    raw_unlabeled_x, raw_unlabeled_y = read_data(filename_unlabeled, "not_available")
    raw_labeled_x, raw_labeled_y = read_data(filename_labeled, label_name)
    raw_labeled = pd.read_csv(filename_labeled)
    labeled_ids = raw_labeled["id"]
    raw_unlabeled = pd.read_csv(filename_unlabeled)
    unlabeled_ids = raw_unlabeled["id"]
    train_x = []
    val_x = []
    val_y = []
    id_dict = {}
    for i in range(len(labeled_ids)):
        id_dict[labeled_ids[i]] = i
    for i in range(len(unlabeled_ids)):
        label_id = unlabeled_ids[i]
        if label_id in id_dict:
            val_x.append(raw_unlabeled_x[i])
            val_y.append(raw_labeled_y[id_dict[label_id]])
        else:
            train_x.append(raw_unlabeled_x[i])
    return np.array(train_x), np.array(val_x), np.array(val_y)

def read_data(filename, label_name):
    raw = pd.read_csv(filename)
    show_features(raw)
    filter_wrapper = FilterWrapper()
    add_filters(filter_wrapper)
    raw_x, raw_y = filter_wrapper.run(raw, label_name)
    print("The shape of x after preprocessing: ", raw_x.shape)
    try:
        print("The shape of y after preprocessing: ", raw_y.shape)
    except:
        print("labels do not exist")
    print("The first row of x: ")
    print(raw_x[0,:])
    try:
        print("The first 3 rows of y: ")
        print(raw_y[:3,:])
    except:
        print("labels do not exist")
    return raw_x, raw_y

def partition_data(x, y, ratio=0.7):
    if len(x) != len(y):
        raise ValueError("Data and label have different length")
    size = len(x)
    train_size = int(ratio * size)
    train_x = x[:train_size]
    train_y = y[:train_size]
    val_x = x[train_size:]
    val_y = y[train_size:]
    return train_x, train_y, val_x, val_y

def add_filters(filter_wrapper):
    # Basic filters
    filter_wrapper.add('danceability', BasicFilterFunc())
    filter_wrapper.add('energy', BasicFilterFunc())
    # filter_wrapper.add('key', BasicFilterFunc())
    filter_wrapper.add('loudness', BasicFilterFunc())
    # filter_wrapper.add('speechiness', BasicFilterFunc())
    # filter_wrapper.add('acousticness', BasicFilterFunc())
    # filter_wrapper.add('instrumentalness', BasicFilterFunc())
    filter_wrapper.add('liveness', BasicFilterFunc())
    # filter_wrapper.add('valence', BasicFilterFunc())
    # filter_wrapper.add('tempo', BasicFilterFunc())
    # filter_wrapper.add('duration_ms', BasicFilterFunc())

    # Shrinking filters
    filter_wrapper.add('danceability', ShrinkFilterFunc())
    filter_wrapper.add('energy', ShrinkFilterFunc())
    # filter_wrapper.add('key', ShrinkFilterFunc())
    filter_wrapper.add('loudness', ShrinkFilterFunc())
    # filter_wrapper.add('speechiness', ShrinkFilterFunc())
    # filter_wrapper.add('acousticness', ShrinkFilterFunc())
    # filter_wrapper.add('instrumentalness', ShrinkFilterFunc())
    filter_wrapper.add('liveness', ShrinkFilterFunc())
    # filter_wrapper.add('valence', ShrinkFilterFunc())
    # filter_wrapper.add('tempo', ShrinkFilterFunc())
    # filter_wrapper.add('duration_ms', ShrinkFilterFunc())

    # Categorical filters
    filter_wrapper.add('artist_era_1', CategoricalOneHotFilterFunc())
    # filter_wrapper.add('artist_era_2', CategoricalOneHotFilterFunc())
    filter_wrapper.add('artist_origin_1', CategoricalOneHotFilterFunc())
    # filter_wrapper.add('artist_origin_2', CategoricalOneHotFilterFunc())
    # filter_wrapper.add('artist_origin_3', CategoricalOneHotFilterFunc())
    # filter_wrapper.add('artist_origin_4', CategoricalOneHotFilterFunc())
    filter_wrapper.add('artist_type_1', CategoricalOneHotFilterFunc())
    # filter_wrapper.add('artist_type_2', CategoricalOneHotFilterFunc())
    filter_wrapper.add('genre_1', CategoricalOneHotFilterFunc())
    # filter_wrapper.add('genre_2', CategoricalOneHotFilterFunc())
    # filter_wrapper.add('genre_3', CategoricalOneHotFilterFunc())
    filter_wrapper.add('tempo_1', CategoricalOneHotFilterFunc())
    # filter_wrapper.add('tempo_2', CategoricalOneHotFilterFunc())
    # filter_wrapper.add('tempo_3', CategoricalOneHotFilterFunc())
    filter_wrapper.add('time_signature', CategoricalOneHotFilterFunc())
    filter_wrapper.add('mode', CategoricalOneHotFilterFunc())
    filter_wrapper.add('key', CategoricalOneHotFilterFunc())
    filter_wrapper.add('mood_1', CategoricalOneHotFilterFunc(verbose=True))
    # filter_wrapper.add('mood_2', CategoricalOneHotFilterFunc())
