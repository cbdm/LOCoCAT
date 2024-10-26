# This script measures the latency for the trained MLPs.

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

seed(0)  # for reproducibility

DATASETS = [
    # name, dataset_path
    ("car-hacking", join(pardir, "datasets", "car-hacking", "models", "start50ms", "dataset.pickle")),
    ("survival-ids", join(pardir, "datasets", "survival-ids", "models", "start50ms", "dataset.pickle")),
    ("syncan", join(pardir, "datasets", "syncan", "models", "start500ms", "dataset.pickle")),
]

EXPORT_PATH = join(pardir, "related_models")
EXPORT_FILE = "mlp3-latencies.csv"

N = 100

results = {}
models = set()

with open(join(EXPORT_PATH, EXPORT_FILE), "a") as file_out:
    file_out.write("model,datapoints,repetitions,total_inferences,total_time,average_time\n")

for (name, data_path) in DATASETS:
    with open(data_path, "rb") as file_in:
        X_train, X_test, _, _ = pickle.load(file_in)

    attacks = [
        attack.reshape(1, -1) for attack in np.concatenate((X_train, X_test))
    ]

    model_name = f"mlp3-{name}.pickle"
    models.add(model_name)
    print("Measuring", model_name)

    with open(join(EXPORT_PATH, model_name), "rb") as file_in:
        clf = pickle.load(file_in)

    start = timer()
    for _ in range(N):
        for a in attacks:
            clf.predict(a)
    end = timer()
    diff = end - start

    print(f"\t\ttotal time elapsed: {diff}s (avg: {diff/(N*len(attacks))})")
    with open(join(EXPORT_PATH, EXPORT_FILE), "a") as file_out:
        file_out.write(f"mlp3-{name},{len(attacks)},{N},{len(attacks) * N},{diff},{diff/(len(attacks) * N)}\n")
