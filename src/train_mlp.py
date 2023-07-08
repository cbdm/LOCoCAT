# This script trains MLPs inspired by the paper by Amato et al.
# Ref: https://doi.org/10.1109/tits.2020.3046974
# In their work, an MLP with 3 hidden layers achieves an average f1-score of 96.8% on the Car-Hacking dataset.

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

seed(0)  # for reproducibility

DATASETS = [
    # name, dataset_path, hidden layers size
    # HLs have a similar setup as the best model in reference paper (HL1 = dim_X - 2, HL2 = HL1 + 2, HL3 = HL3 + 2)
    ("car-hacking", join(pardir, "datasets", "car-hacking", "models", "start50ms", "dataset.pickle"), (23, 25, 27)),
    ("survival-ids", join(pardir, "datasets", "survival-ids", "models", "start50ms", "dataset.pickle"), (23, 25, 27)),
    ("syncan", join(pardir, "datasets", "syncan", "models", "start500ms", "dataset.pickle"), (11, 13, 15)),
]

EXPORT_PATH = join(pardir, "related_models")
print(f"Export path set to '{EXPORT_PATH}', making sure it exists...")
if not exists(EXPORT_PATH):
    print(f"\tattempting to create '{EXPORT_PATH}'...")
    makedirs(EXPORT_PATH)
assert isdir(EXPORT_PATH)
print("\texport path exists.\n")

EXPORT_FILE = "mlp3-score.csv"

with open(join(EXPORT_PATH, EXPORT_FILE), "a") as file_out:
    file_out.write("model,test_score\n")

for (name, path, hl) in DATASETS:
    model_path = join(EXPORT_PATH, f"mlp3-{name}.pickle")
    with open(path, "rb") as file_in:
        X_train, X_test, y_train, y_test = pickle.load(file_in)

    clf = MLPClassifier(solver='adam', alpha=1e-5,
                       hidden_layer_sizes=hl, random_state=1)
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f"{name} test score: {test_score}")
    with open(join(EXPORT_PATH, EXPORT_FILE), "a") as file_out:
        file_out.write(f"mlp3-{name},{test_score}\n")

    with open(model_path, "wb") as file_out:
        pickle.dump(clf, file_out)

    print("Model exported.")
