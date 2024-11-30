# This script trains LSTMs inspired by the paper by Hossain et al.
# Ref: https://doi.org/10.1109/ACCESS.2020.3029307
# In their work, a single-layer 512-units LSTM achieves an accuracy of 99.995% on the Survival-IDS dataset.
# This implementation was based on: https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

import numpy as np
from pandas import read_csv
import dill as pickle
import math
import tensorflow as tf
from random import seed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Nadam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from os.path import join, pardir

DATASETS = [
    # name, path, columns, label_first
    # ("survival-ids", join(pardir, "data", "survival-ids_attacks-all.csv"), [3,4,5,6,7,8,9,10,11,12], False),
    # ("car-hacking", join(pardir, "data", "car-hacking_attack_only.csv"), [3,4,5,6,7,8,9,10,11,12], False),
    ("syncan", join(pardir, "data", "syncan_attack_only.csv"), [1,4,5,6,7], True),
]
EXPORT_PATH = join(pardir, "related_models")
EXPORT_FILE = "lstm-score.csv"

EPOCHS = 50
LEARNING_RATE = 0.000_1
LOOK_BACK = 1
TRAIN_PCT = 0.8

# convert an array of values into a dataset matrix
def create_dataset(dataset, label_first, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        if label_first:
            a = dataset[i:(i+look_back), 1:]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        else:
            a = dataset[i:(i+look_back), :-1]
            dataX.append(a)
            dataY.append(dataset[i + look_back, -1])
    return np.array(dataX), np.array(dataY)


def to_one_hot(Ys):
    _ys = Ys.astype(np.int32)  # int32 to run on pi0
    out = np.zeros((_ys.size, _ys.max() + 1))
    out[np.arange(_ys.size), _ys] = 1
    return out


def onehot_accuracy(y_true, y_pred):
    return sum([1 if np.argmax(t) == np.argmax(p) else 0 for t, p in zip(y_true, y_pred)]) / len(y_true)


def export_models(model, model_name, num_features):
    path = f"{EXPORT_PATH}/{model_name}"

    run_model = tf.function(lambda x: model(x))
    # This is important, let's fix the input size.
    BATCH_SIZE = 1
    STEPS = 1
    INPUT_SIZE = num_features
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], model.inputs[0].dtype))

    # tf.saved_model.save(model, path)
    tf.saved_model.save(model, export_dir=path, signatures=concrete_func)

    converter = tf.lite.TFLiteConverter.from_saved_model(path)

    tflite_model = converter.convert()
    with open(f"{path}.tflite", 'wb') as f:
        f.write(tflite_model)


def export_random_samples(Xs, filepath, num_samples):
    seed(7)
    idx = np.random.randint(Xs.shape[0], size=num_samples)
    with open(filepath, "wb") as file_out:
        pickle.dump(Xs[idx, :, :].astype(np.float32), file_out)


with open(join(EXPORT_PATH, EXPORT_FILE), "a") as file_out:
    file_out.write(f"model,score\n")


for (name, path, cols, label_first) in DATASETS:
    print(f"Creating models for {name}...")
    # fix random seed for reproducibility
    tf.random.set_seed(7)
    seed(7)

    # load the dataset
    print("Loading data...")
    dataframe = read_csv(path, usecols=cols).fillna(0)
    dataset = dataframe.values
    dataset = dataset.astype('float64')

    X, Y = create_dataset(dataset, label_first, LOOK_BACK)
    print(f"{X.shape=}")
    print(f"{Y.shape=}")

    Yhot = to_one_hot(Y)
    print(f"{Yhot.shape=}")

    # split into train and test sets
    train_size = int(len(dataset) * TRAIN_PCT)
    trainX, trainY = X[:train_size, :], Yhot[:train_size, :]
    print(f"{trainX.shape=}")
    print(f"{trainY.shape=}")

    testX, testY = X[train_size:, :], Yhot[train_size:, :]
    print(f"{testX.shape=}")
    print(f"{testY.shape=}")

    print("Exporting data for latency tests...")
    export_random_samples(X, join(EXPORT_PATH, f"lstm-{name}-samples.pickle"), num_samples=1_000)

    # Model setup:
    # create and fit the LSTM network
    model = Sequential()
    model.add(InputLayer(input_shape=(1, trainX.shape[2]), name='input'),)
    model.add(LSTM(512, activation="sigmoid"))
    model.add(Dense(trainY.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Nadam(learning_rate=LEARNING_RATE))
    model.fit(trainX, trainY, epochs=EPOCHS, batch_size=256, verbose=1)

    trainPred = model.predict(trainX)
    print(f"{trainPred.shape=}")
    print(f"{trainY.shape=}")
    print("Train score:", onehot_accuracy(trainY, trainPred))

    testPred = model.predict(testX)
    print(f"{testPred.shape=}")
    print(f"{testY.shape=}")
    score = onehot_accuracy(testY, testPred)
    print("Test score:", score)

    print("Exporting score...")
    with open(join(EXPORT_PATH, EXPORT_FILE), "a") as file_out:
        file_out.write(f"lstm-{name},{score}\n")

    print("Exporting model...")
    export_models(model, model_name=f"lstm-{name}", num_features=trainX.shape[2])
