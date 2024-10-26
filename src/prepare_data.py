from os.path import pardir, join, exists, isdir
import pandas as pd
import numpy as np
import zipfile
import os
from os import remove, listdir
from itertools import product


def prepare_syncan():
    SynCAN_path = join(pardir, "data", "SynCAN")
    assert exists(
        SynCAN_path
    ), "SynCAN dataset doesn't exist. Make sure you have cloned the submodules of this repo."

    result_filepath = join(pardir, "data", "syncan_attack_only.csv")
    assert not exists(result_filepath), f"'{result_filepath}' already exists."

    attack_types = [
        "continuous",  # label==0
        "plateau",  # label==1
        "playback",  # label==2
        "suppress",  # label==3
        "flooding",  # label==4
    ]

    for at in attack_types:
        assert exists(
            join(SynCAN_path, f"test_{at}.zip")
        ), f"Missing data for '{at}' attacks."

    result_df = None
    for i, at in enumerate(attack_types):
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

        # Remove non-attack data.
        print(f"\t- removing non-attack data")
        df = df[df.Label != 0]

        # Change the label so we know which attack type it is.
        print("\t- changing the label")
        df["Label"] = df["Label"].replace(1, i)

        if result_df is None:
            result_df = df
        else:
            result_df = pd.concat([result_df, df], ignore_index=True, sort=False)

    print(f"Creating '{result_filepath}' with all attack data...")
    result_df.to_csv(result_filepath, index_label="Index")

    print(f"Cleaning up unzipped files...")
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
    filename, label, *, timeshift=None, timeshift_constant=10_000_000_000_000
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
    data["Label"] = label
    # Adjusts timestamps from s to ms.
    data["Time"] *= 1_000
    # If needed, adjusts timestamps so different files don't have conflicts.
    if timeshift is not None:
        timechange = timeshift * timeshift_constant
        data["Time"] += timechange
    # Filters only the attack messages.
    attacks = data[data.Flag == "T"]
    # Removes the Flag which is not necesssary anymore and returns the DF.
    attacks = attacks.drop(columns=["Flag"])
    # Fixes types on df and returns final result.
    return _fix_types_CH_IDS(attacks)


def prepare_car_hacking():
    url = "https://ocslab.hksecurity.net/Datasets/car-hacking-dataset"
    CarHacking_dataset = join(pardir, "data", "9) Car-Hacking Dataset.zip")
    assert exists(
        CarHacking_dataset
    ), f"Car-Hacking dataset doesn't exist. Make sure you have downloaded it from: {url}"

    result_filepath = join(pardir, "data", "car-hacking_attack_only.csv")
    assert not exists(result_filepath), f"'{result_filepath}' already exists."

    CarHacking_path = CarHacking_dataset[: -len(".zip")]
    if not exists(CarHacking_path):
        with zipfile.ZipFile(CarHacking_dataset, "r") as zip_ref:
            zip_ref.extractall(CarHacking_path)
    assert exists(CarHacking_path) and isdir(
        CarHacking_path
    ), f"Could not find directory '{CarHacking_path}'."

    attack_types = [
        "DoS",  # label==0
        "Fuzzy",  # label==1
        "gear",  # label==2
        "RPM",  # label==3
    ]

    for at in attack_types:
        assert exists(
            join(CarHacking_path, f"{at}_dataset.csv")
        ), f"Missing data for '{at}' attacks."

    result_df = None
    for label, at in enumerate(attack_types):
        filename = f"{at}_dataset.csv"
        print(f"Processing '{filename}'; assigning {label=}")
        new_df = _parse_CH_IDS_attacks(
            join(CarHacking_path, filename), label, timeshift=label
        )
        if result_df is None:
            result_df = new_df
        else:
            result_df = pd.concat([result_df, new_df])

    result_df = result_df.reset_index(drop=True)
    result_df.to_csv(result_filepath, index_label="Index")


def prepare_survival_ids():
    url = "https://ocslab.hksecurity.net/Datasets/survival-ids"
    SurvivalIDS_dataset = join(
        pardir, "data", "20) Survival Analysis Dataset for automobile IDS.zip"
    )
    assert exists(
        SurvivalIDS_dataset
    ), f"Survival-IDS dataset doesn't exist. Make sure you have downloaded it from: {url}"

    result_filepath = join(pardir, "data", "survival-ids_attacks")
    assert not exists(
        f"{result_filepath}-all.csv"
    ), f"'{result_filepath}-all.csv' already exists."

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
        "Flooding",  # label==0,
        "Fuzzy",  # label==1,
        "Malfunction",  # label==2
    ]

    dfs = {}
    for i, (car, at) in enumerate(product(cars, attack_types)):
        car_subdir = join(SurvivalIDS_subdir, car)
        filename = [f for f in listdir(car_subdir) if f.startswith(at)][0]
        label = i % len(cars)
        print(f"Processing '{filename}'; assigning {label=}")
        new_df = _parse_CH_IDS_attacks(join(car_subdir, filename), label, timeshift=i)
        if car not in dfs:
            dfs[car] = new_df
        else:
            dfs[car] = pd.concat([dfs[car], new_df])

    result_df = None
    for car in dfs:
        dfs[car] = dfs[car].reset_index(drop=True)
        dfs[car].to_csv(f"{result_filepath}-{car}.csv", index_label="Index")
        if result_df is None:
            result_df = dfs[car]
        else:
            result_df = pd.concat([result_df, dfs[car]])

    result_df = result_df.reset_index(drop=True)
    result_df.to_csv(f"{result_filepath}-all.csv", index_label="Index")

def parse_canlog(filename, label=None):

    # reads in a file and the corresponding type of attack
    # makes columns for time, attack id, and the attack data
    data = pd.read_csv(
        filename,
        header=None,
        names = [
            "Time",
            "Network",
            "ID",
            "Message Data",
            "Extra"
        ],
        sep = "[ |#]",
        engine = "python",
        usecols = ["Time", "ID", "Message Data"],
        dtype = {"Message Data" : str, "ID": str},
        nrows = 1
    )

    # makes a label column to specify the type of attack``
    if label is not None:
        data["Label"] = label

    # turns the attack data column into eight separate columns (one for each signal)
    # converts the hexademical to decimal values
    for i in range(8):
        data[f"DATA{i}"] = data["Message Data"].str.slice(i*2, (i*2)+2).apply(hex_to_int)
    data = data.drop(columns=["Message Data"])

    # strips the parenthases away from the time label
    data["Time"] = data["Time"].apply(lambda x: x[1:-1]).astype(np.float64)

    return data

def hex_to_int(hex_string):
    try:
        return int(hex_string, 16)
    except (ValueError, TypeError):
        return 0

def prepare_canlog():
    url =  "https://bitbucket.org/brooke-lampe/can-log/src/master/"
    canlog_dataset = join(pardir, "data", "can-log")
    assert exists(
        canlog_dataset
    ), f"can-log dataset doesn't exist. Make sure you have downloaded it from: {url}"

    result_filepath = join(pardir, "data", "can-log.csv")
    assert not exists(result_filepath), f"{result_filepath} already exists."

    attack_types = [
        "DoS-attacks", # label==0
        "fuzzing-attacks", # label==1
        "gear-attacks", # label==2
        "interval-attacks", # label==3
        "rpm-attacks", # label==4
        "speed-attacks", # label==5
        "standstill-attacks", # label==6
        "systematic-attacks", # label==7
    ]

    datasets = [
        "2011-chevrolet-impala",
        "2011-chevrolet-traverse",
        "2016-chevrolet-silverado", 
        "2017-subaru-forester"
    ]

    for ds in datasets:
        for at in attack_types:
            assert exists(
                join(pardir, "data", "can-log", ds, at)
            ), f"Missing data for '{at}' attacks for {ds}."

    result_df = None
    for label, at in enumerate(attack_types):
        for ds in datasets:
            dir = fr"{pardir}/data/can-log/{ds}/{at}"
            for file in os.listdir(dir):
                print(fr"Processing {dir + "/" + file}; assigning label {label}...")
                new_df = parse_canlog(fr"{dir}/{file}", label)
                    
                if result_df is None:
                    result_df = new_df
                else:
                    result_df = pd.concat([result_df, new_df])

    result_df = result_df.reset_index(drop=True)
    result_df.to_csv(result_filepath, index_label="Index")

def prepare_can_mirgu():
    url = "https://github.com/sampathrajapaksha/CAN-MIRGU"
    can_mirgu_dataset = join(pardir, "data", "CAN-MIRGU/Sample dataset/Attack")
    assert exists(
        can_mirgu_dataset
    ), f"CAN-MIRGU dataset doesn't exist. Make sure you have downloaded it from {url}"

    result_filepath = join(pardir, "data", "can-mirgu.csv")
    assert not exists(result_filepath), f"{result_filepath} already exists"

    attack_types = [
        "Masquerade_attacks", # label==0
        "Suspension_attacks", # label==1
        "Real_attacks", # label==2
    ]

    for at in attack_types:
        assert exists(
            can_mirgu_dataset + f"/{at}"
        ), f"Missing data for '{at}' attacks."

    result_df = None
    for label, at in enumerate(attack_types):
        for file in os.listdir(can_mirgu_dataset + f"/{at}"):
            path = can_mirgu_dataset + "/" + at + "/" + file
            if file.endswith(".log"):
                print(fr"Processing {path}; assigning label {label}...")
                new_df = parse_canlog(path, label)

                if result_df is None:
                    result_df = new_df
                else:
                    result_df = pd.concat([result_df, new_df])
            elif file.endswith(".zip"):
                new_path = can_mirgu_dataset + "/" + at + "/extracted_files"
                print(fr"Extracting files into {new_path}...")
                with zipfile.ZipFile(path, "r") as zip:
                    zip.extractall(new_path)
                for extracted in os.listdir(new_path):
                    if extracted.endswith(".log"):
                        print(fr"Processing {new_path + "/" + extracted}; assigning label {label}...")
                        new_df = parse_canlog(new_path + "/" + extracted, label)

                        if result_df is None:
                            result_df = new_df
                        else:
                            result_df = pd.concat([result_df, new_df])
    
    result_df = result_df.reset_index(drop=True)
    result_df.to_csv(result_filepath, index_label="Index")

if __name__ == "__main__":
    prepare_syncan()
    prepare_car_hacking()
    prepare_survival_ids()
    prepare_canlog()
    prepare_can_mirgu()