import pandas as pd
from .visualization import show_features
from .preprocess import FilterWrapper
from .filters import *

def read_data(filename):
    raw = pd.read_csv(filename)
    show_features(raw)
    filter_wrapper = FilterWrapper()
    add_filters(filter_wrapper)
    raw_x, raw_y = filter_wrapper.run(raw, 'mood_1')
    print("The shape of x after preprocessing: ", raw_x.shape)
    print("The shape of y after preprocessing: ", raw_y.shape)
    print("The first 3 rows of x: ")
    print(raw_x[:3,:])
    print("The first 3 rows of y: ")
    print(raw_y[:3,:])
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
    filter_wrapper.add('danceability', BasicFilterFunc())
    filter_wrapper.add('energy', BasicFilterFunc())
    filter_wrapper.add('key', BasicFilterFunc())
    filter_wrapper.add('loudness', BasicFilterFunc())
    filter_wrapper.add('mode', BasicFilterFunc())
    filter_wrapper.add('speechiness', BasicFilterFunc())
    filter_wrapper.add('acousticness', BasicFilterFunc())
    filter_wrapper.add('instrumentalness', BasicFilterFunc())
    filter_wrapper.add('liveness', BasicFilterFunc())
    filter_wrapper.add('valence', BasicFilterFunc())
    filter_wrapper.add('tempo', BasicFilterFunc())
    filter_wrapper.add('duration_ms', BasicFilterFunc())
    filter_wrapper.add('time_signature', BasicFilterFunc())
    filter_wrapper.add('artist_era_1', CategoricalOneHotFilterFunc())
    filter_wrapper.add('artist_era_2', CategoricalOneHotFilterFunc())
    filter_wrapper.add('artist_origin_1', CategoricalOneHotFilterFunc())
    filter_wrapper.add('artist_origin_2', CategoricalOneHotFilterFunc())
    filter_wrapper.add('artist_origin_3', CategoricalOneHotFilterFunc())
    filter_wrapper.add('artist_origin_4', CategoricalOneHotFilterFunc())
    filter_wrapper.add('artist_type_1', CategoricalOneHotFilterFunc())
    filter_wrapper.add('artist_type_2', CategoricalOneHotFilterFunc())
    filter_wrapper.add('genre_1', CategoricalOneHotFilterFunc())
    filter_wrapper.add('genre_2', CategoricalOneHotFilterFunc())
    filter_wrapper.add('genre_3', CategoricalOneHotFilterFunc())
    filter_wrapper.add('mood_1', CategoricalOneHotFilterFunc())
    filter_wrapper.add('tempo_1', CategoricalOneHotFilterFunc())
    filter_wrapper.add('tempo_2', CategoricalOneHotFilterFunc())
    filter_wrapper.add('tempo_3', CategoricalOneHotFilterFunc())
