import numpy as np
import pandas as pd
import helper.visualization as viz
import helper.preprocess as prep
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten

def read_data(filename):
    raw = pd.read_csv(filename)
    viz.show_features(raw)
    filter_wrapper = prep.FilterWrapper()
    raw_dataset = filter_wrapper.run(raw)
    return raw_dataset

def partition_data(data):
    raise ValueError("not implemented")

def select_feature(data):
    return data

def sample_train(x, y, size=50):
    raise ValueError("not implemented")

def main():
    print("starting main process ...")
    train_raw = read_data("data/train_songs.csv")
    test_raw = read_data("data/test_songs.csv")
    data_selected = select_feature(data_all)
    train_x, train_y = partition_data(data=train_raw)
    val_x, val_y = partition_data(data=test_raw)
    iter_cnt = 10
    model_list = os.listdir("model")
    model = None
    if "saved_model.h5" in model_list:
        print("found previous model, recovering ...")
        model = load_model('model/saved_model.h5')
        print("previous model loaded")
    else:
        print("no previous model found, constructing a new one ...")
        model = Sequential()
        feature_cnt = train_x.shape[1]
        class_cnt = train_y.shape[1]
        model.add(Dense(32, activation='relu', input_shape=(feature_cnt,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(class_cnt, activation='softmax'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        print("finished constructing new model")
    for i in range(iter_cnt):
        print("executing iteration ", str(i), " out of ", str(iter_cnt), " ... ")
        sample_x, sample_y = sample_train(x=train_x, y=train_y, size=50)
        model.fit(x=sample_x, y=sample_y, batch_size=50, epochs=10, verbose=1)
        print("finished this round of training, saving model snapshot ...")
        model.save('model/saved_model.h5')
    print("start testing ...")
    score = model.evaluate(x=val_x, y=val_y, batch_size=50, verbose=1)
    print("final testing score is ", score)

if __name__ == "__main__":
    try:
        main()
    except ValueError as err:
        for e in err.args:
            print(e)
