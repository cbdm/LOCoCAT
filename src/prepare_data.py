import argparse
import json
import os
import re
import zipfile
from itertools import product
from os import listdir, remove
from os.path import basename, exists, isdir, join, pardir

import numpy as np
import pandas as pd


def prepare_syncan():
    SynCAN_path = join(pardir, "data", "SynCAN")
    assert exists(
        SynCAN_path
    ), "SynCAN dataset doesn't exist. Make sure you have cloned the submodules of this repo."

    result_filepath = join(pardir, "data", f"syncan{RESULT_FILE_SUFFIX}.csv")
    assert not exists(result_filepath), f"'{result_filepath}' already exists."

    attack_types = [
        "continuous",  # label==1
        "plateau",  # label==2
        "playback",  # label==3
        "suppress",  # label==4
        "flooding",  # label==5
    ]

    for at in attack_types:
        assert exists(
            join(SynCAN_path, f"test_{at}.zip")
        ), f"Missing data for '{at}' attacks."

    result_df = None
    for i, at in enumerate(attack_types):
        label = i + 1
        print(f"Processing test_{at}.zip")

        # Expand the zip file.
        print(
            f"\t- expanding the file inside {SynCAN_path}...",
        )
        with zipfile.ZipFile(join(SynCAN_path, f"test_{at}.zip"), "r") as zip_ref:
            zip_ref.extractall(SynCAN_path)
        print("\t  done!")

        # Read the data.
        print("\t- reading the data")
        df = pd.read_csv(join(SynCAN_path, f"test_{at}.csv"), header=0)

        # Change the label so we know which attack type it is.
        print("\t- changing the label")
        df["Label"] = df["Label"].replace(1, label)

        # Remove normal data if appropriate.
        print(f"{REMOVE_NON_ATTACK_MESSAGES=}")
        if REMOVE_NON_ATTACK_MESSAGES:
            df = df[df["Label"] > 0]

        if result_df is None:
            result_df = df
        else:
            result_df = pd.concat([result_df, df], ignore_index=True, sort=False)

    print("Renaming fields to match other datasets...")
    result_df = result_df.rename(
        columns={
            "Signal1_of_ID": "DATA0",
            "Signal2_of_ID": "DATA1",
            "Signal3_of_ID": "DATA2",
            "Signal4_of_ID": "DATA3",
        }
    )

    print("Adding extra signal columns to match other datasets...")
    for i in range(4, 8):
        result_df[f"DATA{i}"] = np.nan

    # If needed, add a column with attack block numbers.
    # Check `analysis/find-blocks.py` for how we got to this logic.
    if ADD_ATTACK_BLOCK_LABELS:
        attack_split_factor = 1_000
        result_df["Start of Block"] = (
            result_df.Time.diff().fillna(attack_split_factor) // attack_split_factor
        ) > 0
        result_df["Block Number"] = result_df["Start of Block"].cumsum()
        result_df = result_df.drop(columns=["Start of Block"])

    print(f"Creating '{result_filepath}' with all attack data...")
    result_df.to_csv(result_filepath, index_label="Index")

    print("Cleaning up unzipped files...")
    for at in attack_types:
        filepath = join(SynCAN_path, f"test_{at}.csv")
        print(f"\t- removing '{filepath}'...")
        remove(filepath)

    print("All done :)")


def _fill_CH_IDS_rows(df):
    """Add 0-filled columns on rows that have DLC < 8."""
    assert df.DLC.min() == df.DLC.max()
    num_signals = df.DLC.min()

    if num_signals == 8:
        # Already filled df.
        return df

    missing_signals = 8 - num_signals
    columns = [f"DATA{i}" for i in range(num_signals, 8)] + ["Flag"]
    df[columns] = df[columns].shift(axis=1, periods=missing_signals, fill_value="00")
    return df


def _fix_types_CH_IDS(df):
    """Fix types on DataFrame."""
    df["Time"] = df["Time"].astype(np.float64)
    df["ID"] = df["ID"].astype(str)
    df["DLC"] = df["DLC"].astype(np.int8)
    df["Label"] = df["Label"].astype(np.int8)

    # Normalize values from the signals to be from 0 to 1, instead of 00 to ff.
    for i in range(0, 8):
        df[f"DATA{i}"] = df[f"DATA{i}"].apply(int, base=16)
        df[f"DATA{i}"] /= 255

    return df


def _parse_CH_IDS_attacks(
    filename,
    label,
    *,
    timeshift=None,
    timeshift_constant=10_000_000_000_000,
):
    # Time : recorded time (s)
    # ID : identifier of CAN message in HEX (ex. 043f)
    # DLC : number of data bytes, from 0 to 8
    # DATA0~7 : data value (byte)
    # Flag : T or R, T represents injected message while R represents normal message
    data = pd.read_csv(
        filename,
        header=None,
        names=[
            "Time",
            "ID",
            "DLC",
            "DATA0",
            "DATA1",
            "DATA2",
            "DATA3",
            "DATA4",
            "DATA5",
            "DATA6",
            "DATA7",
            "Flag",
        ],
    )
    # Fill the rows that don't have values for all signals.
    data = data.groupby("DLC", group_keys=False).apply(_fill_CH_IDS_rows)
    # Assigns the given label.
    data["Flag"] = data["Flag"].replace("R", 0)
    data["Flag"] = data["Flag"].replace("T", label)
    data = data.rename(columns={"Flag": "Label"})
    # Adjusts timestamps from s to ms.
    data["Time"] *= 1_000
    # If needed, adjusts timestamps so different files don't have conflicts.
    if timeshift is not None:
        timechange = timeshift * timeshift_constant
        data["Time"] += timechange
    # Fixes types on df and returns final result.
    return _fix_types_CH_IDS(data)


def prepare_car_hacking():
    url = "https://ocslab.hksecurity.net/Datasets/car-hacking-dataset"
    CarHacking_dataset = join(pardir, "data", "9) Car-Hacking Dataset.zip")
    assert exists(
        CarHacking_dataset
    ), f"Car-Hacking dataset doesn't exist. Make sure you have downloaded it from: {url}"

    result_filepath = join(pardir, "data", f"car_hacking{RESULT_FILE_SUFFIX}.csv")
    assert not exists(result_filepath), f"'{result_filepath}' already exists."

    CarHacking_path = CarHacking_dataset[: -len(".zip")]
    if not exists(CarHacking_path):
        with zipfile.ZipFile(CarHacking_dataset, "r") as zip_ref:
            zip_ref.extractall(CarHacking_path)
    assert exists(CarHacking_path) and isdir(
        CarHacking_path
    ), f"Could not find directory '{CarHacking_path}'."

    attack_types = [
        "DoS",  # label==1
        "Fuzzy",  # label==2
        "gear",  # label==3
        "RPM",  # label==4
    ]

    for at in attack_types:
        assert exists(
            join(CarHacking_path, f"{at}_dataset.csv")
        ), f"Missing data for '{at}' attacks."

    result_df = None
    for label, at in enumerate(attack_types):
        set_label = label + 1
        filename = f"{at}_dataset.csv"
        print(f"Processing '{filename}'; assigning {set_label=}")
        new_df = _parse_CH_IDS_attacks(
            join(CarHacking_path, filename), set_label, timeshift=label
        )

        # Remove normal data if appropriate.
        if REMOVE_NON_ATTACK_MESSAGES:
            new_df = new_df[new_df["Label"] > 0]

        if result_df is None:
            result_df = new_df
        else:
            result_df = pd.concat([result_df, new_df])

    # If needed, add a column with attack block numbers.
    # Check `analysis/find-blocks.py` for how we got to this logic.
    if ADD_ATTACK_BLOCK_LABELS:
        attack_split_factor = 1_000
        result_df["Start of Block"] = (
            result_df.Time.diff().fillna(attack_split_factor) // attack_split_factor
        ) > 0
        result_df["Block Number"] = result_df["Start of Block"].cumsum()
        result_df = result_df.drop(columns=["Start of Block"])

    print("Exporting to file...")
    result_df = result_df.reset_index(drop=True)
    result_df.to_csv(result_filepath, index_label="Index")
    print("All done :)")


def prepare_survival_ids():
    url = "https://ocslab.hksecurity.net/Datasets/survival-ids"
    SurvivalIDS_dataset = join(
        pardir, "data", "20) Survival Analysis Dataset for automobile IDS.zip"
    )
    assert exists(
        SurvivalIDS_dataset
    ), f"Survival-IDS dataset doesn't exist. Make sure you have downloaded it from: {url}"

    result_filepath = join(pardir, "data", f"survival_ids{RESULT_FILE_SUFFIX}.csv")
    assert not exists(result_filepath), f"'{result_filepath}' already exists."

    SurvivalIDS_path = SurvivalIDS_dataset[: -len(".zip")]
    if not exists(SurvivalIDS_path):
        with zipfile.ZipFile(SurvivalIDS_dataset, "r") as zip_ref:
            zip_ref.extractall(SurvivalIDS_path, pwd=b"ai.spera!+")
    assert exists(SurvivalIDS_path) and isdir(
        SurvivalIDS_path
    ), f"Could not find directory '{SurvivalIDS_path}'."

    SurvivalIDS_subdir = join(SurvivalIDS_path, "dataset")
    if not exists(SurvivalIDS_subdir):
        with zipfile.ZipFile(join(SurvivalIDS_path, "survival.zip"), "r") as zip_ref:
            zip_ref.extractall(SurvivalIDS_path, pwd=b"ai.spera!+")
    assert exists(SurvivalIDS_subdir) and isdir(
        SurvivalIDS_subdir
    ), f"Could not find directory '{SurvivalIDS_subdir}'."

    cars = ["Sonata", "Soul", "Spark"]
    attack_types = [
        "Flooding",  # label==1,
        "Fuzzy",  # label==2,
        "Malfunction",  # label==3
    ]

    dfs = {}
    for i, (car, at) in enumerate(product(cars, attack_types)):
        car_subdir = join(SurvivalIDS_subdir, car)
        filename = [f for f in listdir(car_subdir) if f.startswith(at)][0]
        label = (i % len(cars)) + 1
        print(f"Processing '{filename}'; assigning {label=}")
        new_df = _parse_CH_IDS_attacks(join(car_subdir, filename), label, timeshift=i)

        # Remove normal data if appropriate.
        if REMOVE_NON_ATTACK_MESSAGES:
            new_df = new_df[new_df["Label"] > 0]

        if car not in dfs:
            dfs[car] = new_df
        else:
            dfs[car] = pd.concat([dfs[car], new_df])

    result_df = None
    for car in dfs:
        dfs[car] = dfs[car].reset_index(drop=True)
        if result_df is None:
            result_df = dfs[car]
        else:
            result_df = pd.concat([result_df, dfs[car]])

    # If needed, add a column with attack block numbers.
    # Check `analysis/find-blocks.py` for how we got to this logic.
    if ADD_ATTACK_BLOCK_LABELS:
        attack_split_factor = 1_000
        result_df["Start of Block"] = (
            result_df.Time.diff().fillna(attack_split_factor) // attack_split_factor
        ) > 0
        result_df["Block Number"] = result_df["Start of Block"].cumsum()
        result_df = result_df.drop(columns=["Start of Block"])

    print("Exporting to file...")
    result_df = result_df.reset_index(drop=True)
    result_df.to_csv(result_filepath, index_label="Index")
    print("All done :)")


def parse_canml_file(
    filename,
    label,
    *,
    timeshift=None,
    timeshift_constant=10_000_000_000_000,
    block_num=None,
):

    # reads in a file and the corresponding type of attack
    # makes columns for time, attack id, and the attack data
    data = pd.read_csv(
        filename,
        header=0,
        engine="python",
        dtype={"data_field": str, "ID": str},
    )

    # makes a label column to specify the type of attack``
    if label is not None:
        data["attack"] = data["attack"].replace(1, label)

    # turns the attack data column into eight separate columns (one for each signal)
    # converts the hexademical to decimal values
    for i in range(8):
        data[f"DATA{i}"] = (
            data["data_field"].str.slice(i * 2, (i * 2) + 2).apply(hex_to_int) / 255
        )
    data = data.drop(columns=["data_field"])

    # If needed, adjusts timestamps so different files don't have conflicts.
    if timeshift is not None:
        timechange = timeshift * timeshift_constant
        data["timestamp"] += timechange

    # If needed, add a block number.
    # The block numbers for can-ml are file-based, so each file has its own block.
    # The caller should pass a unique increasing number for each call to this method.
    # Check `analysis/find-blocks.py` for how we got to this logic.
    if ADD_ATTACK_BLOCK_LABELS:
        data["Block Number"] = block_num

    return data.rename(
        columns={
            "timestamp": "Time",
            "attack": "Label",
            "arbitration_id": "ID",
        }
    )


def hex_to_int(hex_string):
    try:
        return int(hex_string, 16)
    except (ValueError, TypeError):
        return 0


def prepare_canml():
    url = "https://bitbucket.org/brooke-lampe/can-ml/src/master/"
    canlog_dataset = join(pardir, "data", "can-ml")
    assert exists(
        canlog_dataset
    ), f"can-ml dataset doesn't exist. Make sure you have downloaded it from: {url}"

    result_filepath = join(pardir, "data", f"can_ml{RESULT_FILE_SUFFIX}.csv")
    assert not exists(result_filepath), f"{result_filepath} already exists."

    attack_types = [
        "DoS-attacks",  # label==1
        "fuzzing-attacks",  # label==2
        "gear-attacks",  # label==3
        "interval-attacks",  # label==4
        "rpm-attacks",  # label==5
        "speed-attacks",  # label==6
        "standstill-attacks",  # label==7
        "systematic-attacks",  # label==8
    ]

    datasets = [
        "2011-chevrolet-impala",
        "2011-chevrolet-traverse",
        "2016-chevrolet-silverado",
        "2017-subaru-forester",
    ]

    for ds in datasets:
        for at in attack_types:
            assert exists(
                join(pardir, "data", "can-ml", ds, "post-attack-labeled", at)
            ), f"Missing post-attack-labeled data for '{at}' attacks for {ds}."

    result_df = None
    timeshift = 0
    for i, at in enumerate(attack_types):
        label = i + 1
        for ds in datasets:
            dir = join(pardir, "data", "can-ml", ds, "post-attack-labeled", at)
            for file in os.listdir(dir):
                print(
                    rf"Processing {dir + "/" + file}; assigning {label=} with a {timeshift=}..."
                )

                if os.path.isdir(join(dir, file)):
                    print("it's a directory, skipping it...")
                    continue

                new_df = parse_canml_file(
                    filename=join(dir, file),
                    label=label,
                    timeshift=timeshift,
                    block_num=timeshift + 1,
                )
                timeshift += 1

                # Remove normal data if appropriate.
                if REMOVE_NON_ATTACK_MESSAGES:
                    new_df = new_df[new_df["Label"] > 0]

                if result_df is None:
                    result_df = new_df
                else:
                    result_df = pd.concat([result_df, new_df])

    print("Exporting to file...")
    result_df = result_df.reset_index(drop=True)
    result_df.to_csv(result_filepath, index_label="Index")
    print("All done :)")


CAN_MIRGU_BLOCK_NUMBER = 1


def parse_can_mirgu_file(
    attack_metadata,
    path,
    labels,
    *,
    timeshift=None,
    timeshift_constant=10_000_000_000_000,
):
    global CAN_MIRGU_BLOCK_NUMBER

    # Retrieves the attack information for this file from the metadata.
    filename = basename(path)[: -len(".log")]
    attack_info = attack_metadata[filename]
    # Deletes the attack metadata so we can keep track if it has already been processed.
    del attack_metadata[filename]

    # reads in a file and the corresponding type of attack
    # makes columns for time, attack id, and the attack data
    data = pd.read_csv(
        path,
        header=None,
        names=["Time", "Network", "ID", "Message Data", "Label"],
        sep="[ |#]",
        engine="python",
        usecols=["Time", "ID", "Message Data", "Label"],
        dtype={"Message Data": str, "ID": str},
    )

    # Remove non-attack data.
    if REMOVE_NON_ATTACK_MESSAGES:
        data = data[data["Label"] > 0]

    # If needed, assign the appropriate block number for this attack.
    # The attack blocks for CAN-MIRGU are described in its associated metadata file.
    if ADD_ATTACK_BLOCK_LABELS:
        data["Block Number"] = CAN_MIRGU_BLOCK_NUMBER

    # turns the attack data column into eight separate columns (one for each signal)
    # converts the hexademical to decimal values
    for i in range(8):
        data[f"DATA{i}"] = (
            data["Message Data"].str.slice(i * 2, (i * 2) + 2).apply(hex_to_int) / 255
        )
    data = data.drop(columns=["Message Data"])

    # strips the parenthases away from the time label
    data["Time"] = data["Time"].apply(lambda x: x[1:-1]).astype(np.float64)

    # Replace the '1' label with the appropriate attack label for each attack.
    if (
        "injection_interval" in attack_info
        or "suspension_interval" in attack_info
        or "masquerade_interval" in attack_info
    ):
        # There's a single attack block in this file, replace all '1' labels with appropriate label.
        data["Label"] = labels.index(attack_info["attack_technique"]) + 1

        # Increase the block number for the next attack.
        CAN_MIRGU_BLOCK_NUMBER += 1

    else:
        # There are multiple different (and disjoint!) attacks in this file.
        # For each one of them, appropriately assigns its the label.

        for i in range(1, 7):
            # First, find the injection inverval for each attack.
            key = f"injection_interval_{i}"
            start, end = attack_info.get(key, (0, 0))
            if start == end == 0:
                # If empty interval, we've processed all attacks in the current file.
                break

            # Next, find the attack type.
            at_key = (
                "attack_technique"
                if "attack_technique" in attack_info
                else f"attack_technique_{i}"
            )
            at = attack_info[at_key]
            at_label = labels.index(at) + 1

            # Finally, replace '1' labels with the appropriate label for this window.
            data.loc[(data["Time"] >= start) & (data["Time"] <= end), "Label"] = (
                at_label
            )
            # Assign the appropriate block number for this attack.
            if ADD_ATTACK_BLOCK_LABELS:
                data.loc[
                    (data["Time"] >= start) & (data["Time"] <= end), "Block Number"
                ] = CAN_MIRGU_BLOCK_NUMBER

            # Increase the block number for the next attack.
            CAN_MIRGU_BLOCK_NUMBER += 1

    # If needed, adjusts timestamps so different files don't have conflicts.
    if timeshift is not None:
        timechange = timeshift * timeshift_constant
        data["Time"] += timechange

    return data


def prepare_can_mirgu():
    url = "https://github.com/sampathrajapaksha/CAN-MIRGU"
    can_mirgu_dataset = join(pardir, "data", "CAN-MIRGU", "Attack")
    assert exists(
        can_mirgu_dataset
    ), f"CAN-MIRGU dataset doesn't exist. Make sure you have downloaded it from google drive using the link in {url}"

    result_filepath = join(pardir, "data", f"can_mirgu{RESULT_FILE_SUFFIX}.csv")
    assert not exists(result_filepath), f"{result_filepath} already exists"

    attack_metadata_path = join(
        pardir, "data", "CAN-MIRGU", "Attack", "Attacks_metadata.json"
    )
    assert exists(attack_metadata_path), "Missing attack metadata information."
    print(
        f"Make sure you have fixed the '{attack_metadata_path}' file; it has an error in line 467, it should be 'injection_interval_6'."
    )

    # Load attack metadata information so we can find the type of each attack.
    with open(attack_metadata_path, encoding="utf8") as file_in:
        # Remove trailing commas from the metadata file.
        metadata = file_in.read()
        metadata = re.sub(r",[ \t\r\n]+}", "}", metadata)
        metadata = re.sub(r",[ \t\r\n]+\]", "]", metadata)
        # Load the json information.
        attack_metadata = json.loads(metadata)
        # Fix filename mismatches between the dataset and the metadata.
        # As of 2025-06-11, these are the mismatches:
        can_mirgu_mismatches = {
            # "Metadata Entry": "Actual Filename"
            "Steering_angle_replay_attack": "Steering_angle_replay",
            "Fuzzing_valid_IDs_and_DoS_attack": "Fuzzing_valid_IDs_DoS",
            "Reverse_speedometer_and_fuzzing_attack": "Reverse_speedometer_fuzzing_attack",
        }
        for og_name, fixed_name in can_mirgu_mismatches.items():
            attack_metadata[fixed_name] = attack_metadata[og_name]
            del attack_metadata[og_name]

    datasets = [
        "Masquerade_attacks",
        "Suspension_attacks",
        "Real_attacks",
    ]

    attack_techniques = [
        "flam",  # label == 1
        "injecting every 0.02s",  # label == 2
        "injecting every 0.001s",  # labe == 3
        "masquerade",  # label == 4
        "suspension",  # label == 5
    ]

    for ds in datasets:
        assert exists(join(can_mirgu_dataset, ds)), f"Missing data for '{ds}' dataset."

    result_df = None
    timeshift = 0
    for ds in datasets:
        for file in os.listdir(join(can_mirgu_dataset, ds)):
            path = join(can_mirgu_dataset, ds, file)
            if file.endswith(".log"):
                print(f"Processing {path} with {timeshift=}...")
                new_df = parse_can_mirgu_file(
                    attack_metadata,
                    path,
                    attack_techniques,
                    timeshift=timeshift,
                )

                timeshift += 1

                if result_df is None:
                    result_df = new_df
                else:
                    result_df = pd.concat([result_df, new_df])

    assert (
        len(attack_metadata) == 0
    ), "Not all attacks in the metadata file have been parsed."

    print("Exporting to file...")
    result_df = result_df.reset_index(drop=True)
    result_df.to_csv(result_filepath, index_label="Index")
    print("All done :)")


if __name__ == "__main__":
    # Global variables to control whether the output file is being used for block threshold
    # analysis (includes normal data) or for attack classification (excludes normal data).
    global RESULT_FILE_SUFFIX
    global REMOVE_NON_ATTACK_MESSAGES
    global ADD_ATTACK_BLOCK_LABELS

    parser = argparse.ArgumentParser(
        description="This script parses the attack datasets to generate files we can use to classify attack types."
    )
    parser.add_argument(
        "-n",
        "--normal",
        help="include non-attack data for datasets that might be useful",
        action="store_true",
    )
    parser.add_argument(
        "-a",
        "--attack",
        help="include only attack data",
        action="store_true",
    )
    parser.add_argument(
        "-b",
        "--block",
        help="include only attack data and label attack blocks",
        action="store_true",
    )

    args = parser.parse_args()
    options_chosen = len([x for x in args._get_kwargs() if x[1]])
    assert (
        options_chosen == 1
    ), "ERROR: You should choose exactly one option of preparing data."

    if args.block:
        print(
            "Preparing data with only attack data -- filtering out normal data from files -- *and* labeling blocks."
        )
        REMOVE_NON_ATTACK_MESSAGES = True
        ADD_ATTACK_BLOCK_LABELS = True
        RESULT_FILE_SUFFIX = "-blocks"

    elif args.attack:
        print(
            "Preparing data with only attack data -- filtering out normal data from files."
        )

        REMOVE_NON_ATTACK_MESSAGES = True
        ADD_ATTACK_BLOCK_LABELS = False
        RESULT_FILE_SUFFIX = "-attack_only"

    else:
        print("Preparing data that includes normal (non-attack) data.")
        REMOVE_NON_ATTACK_MESSAGES = False
        ADD_ATTACK_BLOCK_LABELS = False
        RESULT_FILE_SUFFIX = "-all"

    # prepare_syncan()
    # prepare_car_hacking()
    # prepare_survival_ids()
    prepare_canml()
    prepare_can_mirgu()
    exit(0)
