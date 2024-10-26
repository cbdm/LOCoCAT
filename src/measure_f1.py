import argparse
from dataset import Dataset
from filters import preprocessing_filters
from random import seed
from os.path import join, pardir, exists, isdir
from os import listdir
import dill as pickle
import itertools
from timeit import default_timer as timer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def measure_f1(models_dir, filename):
    seed(0)  # for reproducibility

    assert exists(models_dir), f"Models directory ('{models_dir}') doesn't exist."
    assert isdir(models_dir), f"Models directory ('{models_dir}') is not a directory."
    filepath = join(models_dir, filename)
    avg_types = ("macro", "micro", "weighted")
    for avg in avg_types:
        fname = f"{filepath}-{avg}.csv"
        assert not exists(fname), f"Output file ('{fname}') already exists."

    subdirs = [f for f in listdir(models_dir) if isdir(join(models_dir, f))]
    print(f"Found the following subdirs inside the models directory: {subdirs}")

    results = {}
    models = set()

    for sd in subdirs:
        sd_path = join(models_dir, sd)
        print(f"Processing models in '{sd}'")

        with open(join(sd_path, "dataset.pickle"), "rb") as file_in:
            _, X_test, _, y_test = pickle.load(file_in)
            print(f"\tdataset loaded")

        for model in listdir(sd_path):
            if model == "dataset.pickle":
                continue

            models.add(model)
            print(f"\tprocessing {model=}")

            with open(join(sd_path, model), "rb") as file_in:
                m = pickle.load(file_in)

            print(f"\t\tmodel loaded")
            print(f"\t\tcalculating f1-scores")
            y_pred = m.predict(X_test)
            results[sd, model] = {
                avg: f1_score(y_test, y_pred, average=avg) for avg in avg_types
            }

    for avg in avg_types:
        out_csv = f"model,{','.join(sorted(subdirs))}\n"
        for m in models:
            out_csv += f"{m}"
            for sd in sorted(subdirs):
                out_csv += f",{results[sd, m][avg]}"
            out_csv += "\n"

        with open(f"{filepath}-{avg}.csv", "w") as file_out:
            file_out.write(out_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script predicts the class for all the data N times and measures the inference time."
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
        help="filename to export the f1-scores",
        default="f1_scores",
    )

    args = parser.parse_args()

    subdirs = [f for f in listdir(args.results_dir) if isdir(join(args.results_dir, f))]
    print("Found following subdirs:", ", ".join(subdirs))
    for sd in subdirs:
        print(f"Measuring f1-score for models inside '{sd}' subdir...")
        measure_f1(join(args.results_dir, sd, "models"), args.filename)
