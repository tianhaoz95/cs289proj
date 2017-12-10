import numpy as np
import os
import helper.config as config
import helper.sampling as sp
import helper.utils as utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD
from sklearn.cluster import KMeans
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def main_sanity_check():
    raw_x, raw_y = utils.read_data("data/data_all_modified.csv", "mood_1")
    train_x, train_y, val_x, val_y = utils.partition_data(x=raw_x, y=raw_y, ratio=0.85)
    dummy_y = val_y.tolist()
    total = len(dummy_y)
    correct = 0
    sample = dummy_y[10]
    for i in range(total):
        if sample.index(1) == dummy_y[i].index(1):
            correct = correct + 1
    accuracy = correct / total
    print("sanity check accuracy: ", accuracy)

def main_dnn_depth():
    depths = [i for i in range(1, 11)]
    neural_cnt = 1000
    test_accuracy = []
    test_losses = []
    train_accuracy = []
    train_losses = []
    raw_x, raw_y = utils.read_data("data/data_all_modified.csv", "mood_1")
    train_x, train_y, val_x, val_y = utils.partition_data(x=raw_x, y=raw_y, ratio=0.85)
    for depth in depths:
        print("running ", depth, " layers ...")
        model = Sequential()
        feature_cnt = train_x.shape[1]
        class_cnt = train_y.shape[1]
        model.add(Dense(neural_cnt, activation='relu', input_shape=(feature_cnt,)))
        for i in range(depth):
            model.add(Dense(neural_cnt, activation='relu'))
            model.add(Dropout(0.5))
        model.add(Dense(class_cnt, activation='softmax'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        history = model.fit(x=train_x, y=train_y, batch_size=200, epochs=20, verbose=1)
        test_score = model.evaluate(x=val_x, y=val_y, verbose=1)
        train_score = model.evaluate(x=train_x, y=train_y, verbose=1)
        print("test score: ", test_score)
        print("train score: ", train_score)
        test_accuracy.append(test_score[1])
        test_losses.append(test_score[0])
        train_accuracy.append(train_score[1])
        train_losses.append(train_score[0])
        del model
    plt.figure()
    plt.plot(depths, test_accuracy, label='testing accuracy')
    plt.plot(depths, train_accuracy, label='training accuracy')
    plt.legend()
    plt.title('training and testing accuracy vs number of layers')
    plt.savefig('plots/accuracy_vs_layer_cnt.png')
    plt.figure()
    plt.plot(depths, test_losses, label='testing loss')
    plt.plot(depths, train_losses, label='training loss')
    plt.legend()
    plt.title('training and testing loss vs number of layers')
    plt.savefig('plots/loss_vs_layer_cnt.png')

def main_grid_search():
    print("starting main grid search process ...")
    neural_cnts = [50 * i for i in range(1, 30)]
    test_accuracy = []
    test_losses = []
    train_accuracy = []
    train_losses = []
    raw_x, raw_y = utils.read_data("data/data_all_modified.csv", "mood_1")
    train_x, train_y, val_x, val_y = utils.partition_data(x=raw_x, y=raw_y, ratio=0.85)
    for neural_cnt in neural_cnts:
        print("running ", neural_cnt, " neural per layer ...")
        model = Sequential()
        feature_cnt = train_x.shape[1]
        class_cnt = train_y.shape[1]
        model.add(Dense(neural_cnt, activation='relu', input_shape=(feature_cnt,)))
        model.add(Dense(neural_cnt, activation='relu'))
        model.add(Dense(neural_cnt, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(neural_cnt, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(neural_cnt, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(neural_cnt, activation='relu'))
        model.add(Dense(neural_cnt, activation='relu'))
        model.add(Dense(class_cnt, activation='softmax'))
        sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        history = model.fit(x=train_x, y=train_y, batch_size=200, epochs=100, verbose=1)
        test_score = model.evaluate(x=val_x, y=val_y, verbose=1)
        train_score = model.evaluate(x=train_x, y=train_y, verbose=1)
        print("test score: ", test_score)
        print("train score: ", train_score)
        test_accuracy.append(test_score[1])
        test_losses.append(test_score[0])
        train_accuracy.append(train_score[1])
        train_losses.append(train_score[0])
        del model
    plt.figure()
    plt.plot(neural_cnts, test_accuracy, label='testing accuracy')
    plt.plot(neural_cnts, train_accuracy, label='training accuracy')
    plt.legend()
    plt.title('training and testing accuracy vs neural counts per layer')
    plt.savefig('plots/accuracy_vs_neural_cnt.png')
    plt.figure()
    plt.plot(neural_cnts, test_losses, label='testing loss')
    plt.plot(neural_cnts, train_losses, label='training loss')
    plt.legend()
    plt.title('training and testing loss vs neural counts per layer')
    plt.savefig('plots/loss_vs_neural_cnt.png')

def main_dnn():
    print("starting main dnn process ...")
    raw_x, raw_y = utils.read_data("data/data_all_modified.csv", "mood_1")
    train_x, train_y, val_x, val_y = utils.partition_data(x=raw_x, y=raw_y, ratio=0.85)
    model_list = os.listdir("model")
    model = None
    train_accuracy = []
    train_loss = []
    test_accuracy = []
    test_loss = []
    model_name = config.model_uid + ".h5"
    model_dir = "model/" + model_name
    if model_name in model_list:
        print("found previous model, recovering ...")
        model = load_model(model_dir)
        print("previous model loaded")
    else:
        print("no previous model found, constructing a new one ...")
        model = Sequential()
        feature_cnt = train_x.shape[1]
        class_cnt = train_y.shape[1]
        model.add(Dense(100, activation='relu', input_shape=(feature_cnt,)))
        model.add(Dense(500, activation='relu'))
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(500, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(class_cnt, activation='softmax'))
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        print("finished constructing new model")
    for i in range(config.iter_cnt):
        print("executing iteration ", str(i+1), " out of ", str(config.iter_cnt), " ... ")
        sample_x, sample_y = sp.sample_train(x=train_x, y=train_y, size=config.sample_size)
        history = model.fit(x=sample_x, y=sample_y, batch_size=config.train_batch, epochs=config.train_epochs, verbose=1)
        train_loss.extend(history.history['loss'])
        train_accuracy.extend(history.history['acc'])
        test_score = model.evaluate(x=val_x, y=val_y, verbose=1)
        print("intermediate testing score: ")
        print(test_score)
        test_loss.append(test_score[0])
        test_accuracy.append(test_score[1])
        print("finished this round of training, saving model snapshot ...")
        model.save(model_dir)
    print("start testing ...")
    score = model.evaluate(x=val_x, y=val_y, verbose=1)
    print("final testing score is: ", score)
    print("try predicting ...")
    pred = model.predict(x=val_x, verbose=1)
    print("first 3 rows of prediction: ")
    np.set_printoptions(precision=3)
    print(pred[:3])
    print("first 3 rows of correct label: ")
    print(val_y[:3])
    print("plotting training loss ...")
    plt.figure()
    plt.plot(train_loss)
    plt.savefig("plots/neural_network_training_loss.png")
    print("printing training accuracy ...")
    plt.figure()
    plt.plot(train_accuracy)
    plt.savefig("plots/neural_network_training_accuracy.png")
    print("printing testing loss")
    plt.figure()
    plt.plot(test_loss)
    plt.savefig("plots/neural_network_testing_loss.png")
    print("printing testing accuracy")
    plt.figure()
    plt.plot(test_accuracy)
    plt.savefig("plots/neural_network_testing_accuracy.png")

def main_kmean():
    print("starting main k means process ...")
    train_x, val_x, val_y = utils.read_kmeans_data("data/data_all_modified.csv", "data/no_label_all.csv", "mood_1")
    class_cnt = val_y.shape[1]
    kmeans = KMeans(n_clusters=class_cnt, random_state=0).fit(X=train_x)
    pred = kmeans.predict(X=val_x)
    print("predicted test set: ")
    print(pred)
    accuracy = utils.eval_kmean(pred=pred, correct=val_y.tolist())
    print("k means accuracy: ", accuracy)

def main_kmean_search():
    print("starting main k means search process ...")
    train_x, val_x, val_y = utils.read_kmeans_data("data/data_all_modified.csv", "data/no_label_all.csv", "mood_1")
    class_cnt = val_y.shape[1]
    fake_y = [0 for i in range(len(train_x))]
    bag_cnt = 50
    bag_size = (len(train_x) - 100) // bag_cnt
    sample_sizes = [bag_size * i for i in range(1, bag_cnt)]
    sample_scores = []
    for sample_size in sample_sizes:
        sample_x, sample_y = sp.sample_train(x=train_x, y=fake_y, size=sample_size)
        kmeans = KMeans(n_clusters=class_cnt, random_state=0).fit(X=sample_x)
        pred = kmeans.predict(X=val_x)
        print("predicted test set: ")
        print(pred)
        accuracy = utils.eval_kmean(pred=pred, correct=val_y.tolist())
        print("k means accuracy: ", accuracy)
        sample_scores.append(accuracy)
    plt.figure()
    plt.plot(sample_sizes, sample_scores)
    plt.title('approximated accuracy vs. sample size')
    plt.savefig('plots/kmeans_accuracy_vs_sample_size.png')

def main_tree():
    print("starting decision tree process ...")
    raw_x, raw_y = utils.read_data("data/data_all_modified.csv", "mood_1")
    train_x, train_y, val_x, val_y = utils.partition_data(x=raw_x, y=raw_y, ratio=0.85)
    model = tree.DecisionTreeClassifier()
    model.fit(X=train_x, y=train_y)
    pred = model.predict(X=train_x)
    print("first row of prediction: ")
    print(pred[0])
    score = model.score(X=val_x, y=val_y)
    print("decision tree score: ", score)

def main_random_forest():
    print("starting random forest process ...")
    raw_x, raw_y = utils.read_data("data/data_all_modified.csv", "mood_1")
    train_x, train_y, val_x, val_y = utils.partition_data(x=raw_x, y=raw_y, ratio=0.85)
    model = RandomForestClassifier(random_state=0)
    model.fit(X=train_x, y=train_y)
    pred = model.predict(X=train_x)
    print("first row of prediction: ")
    print(pred[0])
    score = model.score(X=val_x, y=val_y)
    print("random forest score: ", score)

def main_random_forest_search():
    print("starting decision tree search process ...")
    raw_x, raw_y = utils.read_data("data/data_all_modified.csv", "mood_1")
    train_x, train_y, val_x, val_y = utils.partition_data(x=raw_x, y=raw_y, ratio=0.85)
    levels = [i for i in range(1, 100)]
    train_scores = []
    test_scores = []
    for level in levels:
        model = RandomForestClassifier(max_depth=level, random_state=0)
        model.fit(X=train_x, y=train_y)
        pred = model.predict(X=train_x)
        print("first row of prediction: ")
        print(pred[0])
        test_score = model.score(X=val_x, y=val_y)
        print("decision tree testing score: ", test_score)
        train_score = model.score(X=train_x, y=train_y)
        print("decision tree training score: ", train_score)
        train_scores.append(train_score)
        test_scores.append(test_score)
    plt.figure()
    plt.plot(levels, train_scores, label='training score')
    plt.plot(levels, test_scores, label='testing score')
    plt.legend()
    plt.title('training and testing accuracy vs. tree depth')
    plt.savefig('plots/random_forest_accuracy_vs_depth.png')

def main_tree_search():
    print("starting decision tree search process ...")
    raw_x, raw_y = utils.read_data("data/data_all_modified.csv", "mood_1")
    train_x, train_y, val_x, val_y = utils.partition_data(x=raw_x, y=raw_y, ratio=0.85)
    levels = [i for i in range(1, 100)]
    train_scores = []
    test_scores = []
    for level in levels:
        model = tree.DecisionTreeClassifier(max_depth=level)
        model.fit(X=train_x, y=train_y)
        pred = model.predict(X=train_x)
        print("first row of prediction: ")
        print(pred[0])
        test_score = model.score(X=val_x, y=val_y)
        print("decision tree testing score: ", test_score)
        train_score = model.score(X=train_x, y=train_y)
        print("decision tree training score: ", train_score)
        train_scores.append(train_score)
        test_scores.append(test_score)
    plt.figure()
    plt.plot(levels, train_scores, label='training score')
    plt.plot(levels, test_scores, label='testing score')
    plt.legend()
    plt.title('training and testing accuracy vs. tree depth')
    plt.savefig('plots/decision_tree_accuracy_vs_depth.png')

if __name__ == "__main__":
    try:
        if config.train_mode == "sanity_check":
            main_sanity_check()
        if config.train_mode == "dnn":
            main_dnn()
        if config.train_mode == "kmean":
            main_kmean()
        if config.train_mode == "tree":
            main_tree()
        if config.train_mode == "kmean_search":
            main_kmean_search()
        if config.train_mode == "random_forest":
            main_random_forest()
        if config.train_mode == "random_forest_search":
            main_random_forest_search()
        if config.train_mode == "dnn_depth_search":
            main_dnn_depth()
        if config.train_mode == "tree_search":
            main_tree_search()
        if config.train_mode == "dnn_search":
            main_grid_search()
    except ValueError as err:
        for e in err.args:
            print(e)
