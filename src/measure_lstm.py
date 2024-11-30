# This script measures the latency for the trained LSTMs.

from sklearn.neural_network import MLPClassifier
from dataset import Dataset
from features import syncan_features, ch_ids_features
from filters import preprocessing_filters, start_50ms
import dill as pickle
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from random import seed
from os.path import join, pardir, exists, isdir
from os import makedirs
from itertools import product
from timeit import default_timer as timer
import tflite_runtime.interpreter as tflite

seed(0)  # for reproducibility

MODELS = [
    "car-hacking",
    "survival-ids",
    "syncan",
]

EXPORT_PATH = join(pardir, "related_models")
EXPORT_FILE = "lstm-latencies.csv"

N = 100

results = {}
models = set()

with open(join(EXPORT_PATH, EXPORT_FILE), "a") as file_out:
    file_out.write("model,datapoints,repetitions,total_inferences,total_time,average_time\n")

for model_name in MODELS:
    with open(join(EXPORT_PATH, f"lstm-{model_name}-samples.pickle"), "rb") as file_in:
        Xs = pickle.load(file_in)
        Xs = Xs.reshape((Xs.shape[0], 1, 1, Xs.shape[2]))

    models.add(model_name)
    print("Measuring", model_name)

    interpreter = tflite.Interpreter(model_path=join(EXPORT_PATH, f"lstm-{model_name}.tflite"))

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()

    start = timer()
    for _ in range(N):
        for x in Xs:
            interpreter.set_tensor(input_details[0]['index'], x)
            interpreter.invoke()
    end = timer()
    diff = end - start

    print(f"\t\ttotal time elapsed: {diff}s (avg: {diff/(N*len(Xs))})")
    with open(join(EXPORT_PATH, EXPORT_FILE), "a") as file_out:
        file_out.write(f"lstm-{model_name},{len(Xs)},{N},{len(Xs) * N},{diff},{diff/(len(Xs) * N)}\n")
