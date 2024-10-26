# This script trains DummyClassifiers to generate baseline results.

import argparse
from dataset import Dataset
from filters import preprocessing_filters
from random import seed
from os.path import join, pardir, exists, isdir
from os import listdir
import dill as pickle
import itertools
from timeit import default_timer as timer
from numpy import concatenate
from sklearn.dummy import DummyClassifier

STRATEGIES = sorted(["most_frequent", "prior", "stratified", "uniform"])

# This function measures the training time for each model using the best parameters we found on the search.
def measure_dummy_f1(models_dir, filename):
    seed(0)  # for reproducibility

    assert exists(models_dir), f"Models directory ('{models_dir}') doesn't exist."
    assert isdir(models_dir), f"Models directory ('{models_dir}') is not a directory."
    filepath = join(models_dir, filename)
    assert not exists(filepath), f"Output file ('{filepath}') already exists."

    subdirs = [f for f in listdir(models_dir) if isdir(join(models_dir, f))]
    print(f"Found the following subdirs inside the models directory: {subdirs}")
    first_sd = subdirs[0]
    print(
        f"""Assuming all subdirs have the same number of test datapoints and the y_labels are the same through the different windows.
Using dataset from '{first_sd}' to measure dummy F1."""
    )

    with open(join(models_dir, first_sd, "dataset.pickle"), "rb") as file_in:
        X_train, X_test, y_train, y_test = pickle.load(file_in)

    scores = []
    sizes = []

    for s in STRATEGIES:
        clf = DummyClassifier(strategy=s, random_state=0)
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
        sizes.append(len(pickle.dumps(clf)))

    return clf, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script trains a dummy classifier and measures its weighted f1 score."
    )

    parser.add_argument(
        "-r",
        "--results_dir",
        help="path to the results directory created by the training_models script",
        default=join(pardir, "out"),
    )
    parser.add_argument(
        "-f",
        "--filename",
        help="filename to export results",
        default="dummy-score.csv",
    )

    args = parser.parse_args()

    subdirs = [f for f in listdir(args.results_dir) if isdir(join(args.results_dir, f))]
    print("Found following subdirs:", ", ".join(subdirs))

    out_csv = f"dataset,{','.join(sorted(STRATEGIES))}\n"
    for sd in subdirs:
        print(f"Measuring sizes for models inside '{sd}' subdir...")
        model, scores = measure_dummy_f1(join(args.results_dir, sd, "models"), args.filename)
        out_csv += f"dummy-{sd}," + ",".join([f"{x}" for x in scores]) + "\n"
        with open(join(args.results_dir, f"dummy-{sd}.pickle"), "wb") as file_out:
            pickle.dump(model, file_out)

    with open(join(args.results_dir, args.filename), "w") as file_out:
        file_out.write(out_csv)
