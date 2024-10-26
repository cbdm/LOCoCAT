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


def measure_latency(models_dir, filename, n):
    seed(0)  # for reproducibility

    assert exists(models_dir), f"Models directory ('{models_dir}') doesn't exist."
    assert isdir(models_dir), f"Models directory ('{models_dir}') is not a directory."
    filepath = join(models_dir, filename)
    assert not exists(filepath), f"Output file ('{filepath}') already exists."

    subdirs = [f for f in listdir(models_dir) if isdir(join(models_dir, f))]
    print(f"Found the following subdirs inside the models directory: {subdirs}")

    results = {}
    models = set()

    first_model = True
    attacks = X = None

    for sd in subdirs:
        sd_path = join(models_dir, sd)
        print(f"Processing models in '{sd}'")

        with open(join(sd_path, "dataset.pickle"), "rb") as file_in:
            X_train, X_test, _, _ = pickle.load(file_in)
            attacks = [
                attack.reshape(1, -1) for attack in concatenate((X_train, X_test))
            ]
            print(f"\tdataset loaded with {len(attacks)} attacks")

        for model in listdir(sd_path):
            if model == "dataset.pickle":
                continue

            models.add(model)
            print(f"\tprocessing {model=}")

            with open(join(sd_path, model), "rb") as file_in:
                m = pickle.load(file_in)
            print(f"\t\tmodel loaded")
            print(f"\t\tmeasuring inference time with {n} repetitions")
            start = timer()
            for _ in range(n):
                for a in attacks:
                    m.predict(a)
            end = timer()
            diff = end - start

            print(f"\t\ttotal time elapsed: {diff}s")
            results[sd, model] = [diff, len(attacks)]

    out_csv = (
        f"filters,{','.join(sorted(models))},datapoints,repetitions,total_inferences\n"
    )
    for sd in subdirs:
        out_csv += f"{sd},"
        dps = set()
        for m in sorted(models):
            diff, dp = results[sd, m]
            out_csv += f"{diff},"
            dps.add(dp)
        assert len(dps) == 1, "Length of feature set changed during models inference."
        out_csv += f"{dps.pop()},{n},{dp * n}\n"

    with open(filepath, "w") as file_out:
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
        help="filename to export latency results",
        default="inference_latency.csv",
    )
    parser.add_argument(
        "-n",
        "--n",
        help="number of times each models should be executed",
        default=10,
    )

    args = parser.parse_args()

    subdirs = [f for f in listdir(args.results_dir) if isdir(join(args.results_dir, f))]
    print("Found following subdirs:", ", ".join(subdirs))
    for sd in subdirs:
        print(f"Measuring latency for models inside '{sd}' subdir...")
        measure_latency(
            join(args.results_dir, sd, "models"), args.filename, int(args.n)
        )
