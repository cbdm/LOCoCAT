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


# This function measures the training time for each model using the best parameters we found on the search.
def measure_model_size(models_dir, filename):
    seed(0)  # for reproducibility

    assert exists(models_dir), f"Models directory ('{models_dir}') doesn't exist."
    assert isdir(models_dir), f"Models directory ('{models_dir}') is not a directory."
    filepath = join(models_dir, filename)
    assert not exists(filepath), f"Output file ('{filepath}') already exists."

    subdirs = [f for f in listdir(models_dir) if isdir(join(models_dir, f))]
    print(f"Found the following subdirs inside the models directory: {subdirs}")

    results = {}
    models = set()

    for sd in subdirs:
        sd_path = join(models_dir, sd)
        print(f"Processing models in '{sd}'")

        with open(join(sd_path, "dataset.pickle"), "rb") as file_in:
            X_train, _, y_train, _ = pickle.load(file_in)

        for model in listdir(sd_path):
            if model == "dataset.pickle":
                continue

            models.add(model)
            print(f"\tsizing {model=}")

            with open(join(sd_path, model), "rb") as file_in:
                m = pickle.load(file_in)

            estimator = m.estimator
            estimator.set_params(**m.best_params_)
            estimator.fit(X_train, y_train)
            model_size = len(pickle.dumps(estimator))
            print(f"\ttrained model size: {model_size}B")
            results[sd, model] = model_size

    out_csv = f"filters,{','.join(sorted(models))}\n"
    for sd in subdirs:
        out_csv += f"{sd}"
        for m in sorted(models):
            out_csv += f",{results[sd, m]}"
        out_csv += "\n"

    with open(filepath, "w") as file_out:
        file_out.write(out_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script trains the models using the best params found on the grid search."
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
        default="model_sizes.csv",
    )

    args = parser.parse_args()

    subdirs = [f for f in listdir(args.results_dir) if isdir(join(args.results_dir, f))]
    print("Found following subdirs:", ", ".join(subdirs))
    for sd in subdirs:
        print(f"Measuring sizes for models inside '{sd}' subdir...")
        measure_model_size(join(args.results_dir, sd, "models"), args.filename)
