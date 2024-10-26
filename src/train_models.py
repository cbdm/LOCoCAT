import argparse
import numpy as np
import pandas as pd
from typing import Callable, Optional, List, Tuple
from dataclasses import dataclass
from os.path import join, pardir, exists, isdir
from os import makedirs
from sklearn.model_selection import train_test_split
import dill as pickle
from dataset import Dataset
from features import syncan_features, ch_ids_features
from filters import preprocessing_filters, training_filters
from models import models
from random import seed

# Function to train different models on the same data applying different sets of filters.
def explore_models(
    *, models_to_use, export_path, results_filename, dataset, filters_set, features_set
):
    assert len(filters_set) == len(features_set)

    if export_path:
        print(f"Export path set to '{export_path}', making sure it exists...")
        if not exists(export_path):
            print(f"\tattempting to create '{export_path}'...")
            makedirs(export_path)
        assert isdir(export_path)
        print("\texport path created.\n")

    results = {}
    results_path = ""
    if results_filename:
        assert (
            export_path
        ), f"Results filename set ('{results_filename}') but export path is not set."
        assert exists(
            export_path
        ), f"Export path ('{export_path}') does not exist or could not be created."
        assert isdir(
            export_path
        ), f"Export path ('{export_path}') exists but is not a directory."
        results_path = join(export_path, results_filename)
        assert not exists(
            results_path
        ), f"Results file ('{results_path}') already exists."
        print(f"Results path set to '{results_path}'")

    for filters, features in zip(filters_set, features_set):
        print(f"Using {filters=}, {features=}\n")
        cur_path = ""
        filter_str = "-".join([f.name for f in filters]) if filters else "raw"
        if export_path:
            cur_path = join(export_path, filter_str)
            print(
                f"\tcreating '{cur_path}' to save models trained with current filters."
            )
            if not exists(cur_path):
                makedirs(cur_path)
            assert isdir(cur_path)

        dataset_path = join(cur_path, "dataset.pickle")
        X_train = X_test = y_train = y_test = None
        if exists(dataset_path):
            with open(dataset_path, "rb") as file_in:
                X_train, X_test, y_train, y_test = pickle.load(file_in)

        else:
            X, y = dataset.create_feature_set(filters=filters, features=features)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=0, shuffle=True
            )
            with open(dataset_path, "wb") as file_out:
                pickle.dump(
                    (X_train, X_test, y_train, y_test),
                    file_out,
                )

        for model, gs in models_to_use.items():
            print(f"\ttraining {model=}...")

            gs.fit(X_train, y_train)

            train_acc = gs.score(X_train, y_train)
            test_acc = gs.score(X_test, y_test)

            print("\t\tBest params:", gs.best_params_)
            print("\t\tTrain accuracy:", train_acc)
            print("\t\tTest accuracy:", test_acc)

            if results_path:
                results[filter_str, model] = test_acc

            if cur_path:
                model_path = join(cur_path, f"{model}.pickle")
                print(f"\texporting model to '{model_path}'")
                assert not exists(model_path), f"File ('{model_path}') already exists!"
                with open(model_path, "wb") as file_out:
                    pickle.dump(gs, file_out)
                print("\tdone!")

    if results_path:
        print(f"Generating csv results...")
        filters = sorted({f for f, _ in results})
        models = sorted({m for _, m in results})
        out_csv = "model," + ",".join(filters) + "\n"
        for m in models:
            out_csv += m
            for f in filters:
                out_csv += f",{results[f, m]}"
            out_csv += "\n"

        print(f"Exporting csv results...")
        with open(results_path, "w") as file_out:
            file_out.write(out_csv)

        print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script explores the models' parameters to find the best results."
    )
    parser.add_argument(
        "-a",
        "--all",
        help="train models for all 3 datasets",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--syncan",
        help="train models using the SynCAN dataset",
        action="store_true",
    )
    parser.add_argument(
        "-c",
        "--carhacking",
        help="train models using the Car-Hacking dataset",
        action="store_true",
    )
    parser.add_argument(
        "-i",
        "--survivalids",
        help="train models using the Survival-IDS dataset",
        action="store_true",
    )

    args = parser.parse_args()

    options_chosen = len([x for x in args._get_kwargs() if x[1]])
    all_chosen = args.all or options_chosen > 2

    if not options_chosen:
        print("No option chosen; please use --help to see available options.")
        exit(0)

    if all_chosen:
        print("Training for all datasets.")

    if all_chosen or args.syncan:
        seed(0)  # for reproducibility
        # Pre-filtered version of the SynCAN_dataset; contains only attack messages, and attack labels go from 0 to 4.
        # If this file doesn't exist, you can run the "prepare_data.py" script.
        SynCANData = Dataset(
            name="SynCAN Data",
            path=join(pardir, "data", "syncan_attack_only.csv"),
            filters=preprocessing_filters,
            attack_split_factor=500,
        )
        # Use the imported filter- and feature-sets to explore different params for classifier models.
        explore_models(
            models_to_use=models,
            export_path=join(pardir, "out", "syncan", "models"),
            results_filename="test_accuracy.csv",
            dataset=SynCANData,
            filters_set=training_filters,
            features_set=[syncan_features] * len(training_filters),
        )

    if all_chosen or args.carhacking:
        seed(0)  # for reproducibility
        # Pre-filtered version of the Car-Hacking dataset; contains only attack messages, and attack labels go from 0 to 3.
        # If this file doesn't exist, you can run the "prepare_data.py" script.
        CarHackingData = Dataset(
            name="Car-Hacking Data",
            path=join(pardir, "data", "car-hacking_attack_only.csv"),
            filters=preprocessing_filters,
            attack_split_factor=1_000,
        )
        # Use the imported filter- and feature-sets to explore different params for classifier models.
        explore_models(
            models_to_use=models,
            export_path=join(pardir, "out", "car-hacking", "models"),
            results_filename="test_accuracy.csv",
            dataset=CarHackingData,
            filters_set=training_filters,
            features_set=[ch_ids_features] * len(training_filters),
        )

    if all_chosen or args.survivalids:
        seed(0)  # for reproducibility
        # Pre-filtered version of the Survival-IDS dataset; contains only attack messages, and attack labels go from 0 to 2.
        # If this file doesn't exist, you can run the "prepare_data.py" script.
        SurvivalIDSData = Dataset(
            name="Survival-IDS Data",
            path=join(pardir, "data", "survival-ids_attacks-all.csv"),
            filters=preprocessing_filters,
            attack_split_factor=500,
        )
        # Use the imported filter- and feature-sets to explore different params for classifier models.
        explore_models(
            models_to_use=models,
            export_path=join(pardir, "out", "survival-ids", "models"),
            results_filename="test_accuracy.csv",
            dataset=SurvivalIDSData,
            filters_set=training_filters,
            features_set=[ch_ids_features] * len(training_filters),
        )
