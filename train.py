import numpy as np
import pandas as pd
import helper.visualization as viz
import helper.preprocess as prep
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

def read_data(filename):
    raw = pd.read_csv(filename)
    viz.show_features(raw)
    filter_wrapper = prep.FilterWrapper()
    train_x = filter_wrapper.run(raw)
    return train_x

def get_labels():
    raise ValueError("not implemented")

def partition_data(data, labels, test_ratio):
    raise ValueError("not implemented")

def select_feature(data):
    raise ValueError("not implemented")

def main():
    print("starting train ...")
    data_all = read_data("data/train_songs.csv")
    labels = get_labels()
    data_selected = select_feature(data_all)
    train_x, train_y, test_x, test_y = partition_data(data=data, labels=labels, test_ratio=0.3)
    model = Sequential()
    feature_cnt = train_x.shape[1]
    class_cnt = 10
    model.add(Dense(32, input_shape=(feature_cnt,)))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(class_cnt))
    model.add(Activation('softmax'))

if __name__ == "__main__":
    try:
        main()
    except ValueError as err:
        for e in err.args:
            print(e)
