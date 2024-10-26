import dill as pickle
import numpy as np
from os import makedirs
from os.path import join, pardir, exists, isdir
from sklearn.model_selection import train_test_split
from dataset import Dataset
from features import syncan_features, ch_ids_features
from filters import preprocessing_filters, training_filters

# Function to generate pre-created datasets for training/inference on the pi0.
def create_datasets(*, export_path, dataset, filters_set, features_set):
    assert len(filters_set) == len(features_set)

    if export_path:
        print(f"Export path set to '{export_path}', making sure it exists...")
        if not exists(export_path):
            print(f"\tattempting to create '{export_path}'...")
            makedirs(export_path)
        assert isdir(export_path)
        print("\texport path created.\n")

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
        if not exists(dataset_path):
            X, y = dataset.create_feature_set(filters=filters, features=features)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=0, shuffle=True
            )
            with open(dataset_path, "wb") as file_out:
                pickle.dump(
                    (
                        X_train.astype(np.float64),
                        X_test.astype(np.float64),
                        y_train.astype(np.int32),
                        y_test.astype(np.int32),
                    ),
                    file_out,
                )


if __name__ == "__main__":
    SynCANData = Dataset(
        name="SynCAN Data",
        path=join(pardir, "data", "syncan_attack_only.csv"),
        filters=preprocessing_filters,
        attack_split_factor=500,
    )
    create_datasets(
        export_path=join(pardir, "datasets", "syncan", "models"),
        dataset=SynCANData,
        filters_set=training_filters,
        features_set=[syncan_features] * len(training_filters),
    )

    CarHackingData = Dataset(
        name="Car-Hacking Data",
        path=join(pardir, "data", "car-hacking_attack_only.csv"),
        filters=preprocessing_filters,
        attack_split_factor=1_000,
    )
    create_datasets(
        export_path=join(pardir, "datasets", "car-hacking", "models"),
        dataset=CarHackingData,
        filters_set=training_filters,
        features_set=[ch_ids_features] * len(training_filters),
    )

    SurvivalIDSData = Dataset(
        name="Survival-IDS Data",
        path=join(pardir, "data", "survival-ids_attacks-all.csv"),
        filters=preprocessing_filters,
        attack_split_factor=500,
    )
    create_datasets(
        export_path=join(pardir, "datasets", "survival-ids", "models"),
        dataset=SurvivalIDSData,
        filters_set=training_filters,
        features_set=[ch_ids_features] * len(training_filters),
    )
