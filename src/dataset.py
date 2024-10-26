import pandas as pd
import numpy as np
from typing import List, Tuple
from filters import Filter
from features import Feature


class Dataset(object):
    def __init__(
        self,
        name: str,
        path: str,
        filters: List[Filter],
        attack_split_factor: int,
    ):
        self._name = name
        self._path = path
        self._dataset_filters = filters
        self._attack_split_factor = attack_split_factor
        self._data = None

    def _load(self) -> None:
        df = pd.read_csv(self._path)

        print(f"Loaded dataset from '{self._path}'")

        # Filter the dataset if necessary.
        for f in self._dataset_filters:
            print(f"Filtering dataset with with '{f.name=}'")
            df = f.func(df)

        # Find the start of all attack blocks.
        block_starts = (
            df.Time.diff().fillna(self._attack_split_factor)
            // self._attack_split_factor
        )
        block_starts = block_starts[block_starts > 0]

        # Group all messages of the same block.
        blocks = (
            pd.DataFrame(block_starts)
            .reset_index()
            .reset_index()
            .drop(columns=["Time"])
            .rename(columns={"index": "Index", "level_0": "Block"})
        )
        df = pd.merge_asof(df, blocks, on="Index")
        self._data = df.groupby("Block")

        # Make sure each block has only one attack label.
        min_label = self._data.apply(lambda x: x.Label.min())
        max_label = self._data.apply(lambda x: x.Label.max())
        assert (
            min_label.to_list() == max_label.to_list()
        ), "Multiple attack types in the same block!"

    def get_data(self) -> pd.DataFrame:
        if self._data is None:
            self._load()
        return self._data

    def create_feature_set(
        self,
        *,
        filters: List[Filter] = [],
        features: List[Feature] = [],
        one_hot_label: bool = False,
        fillna_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.array]:
        df = self.get_data()
        frames = []

        # Filter the dataset if necessary.
        for i, f in enumerate(filters):
            print(f"Filtering with '{f.name=}' ({i+1} / {len(filters)})")
            df = df.apply(f.func).reset_index(level=0, drop=True).groupby("Block")

        # Calculate the features.
        for i, feat in enumerate(features):
            print(f"Calculating '{feat.name=}' ({i+1} / {len(features)})")
            frames.append(pd.DataFrame({feat.name: df.apply(feat.func)}))

        # Get the min label since we know it's unique for each block.
        y = df.apply(lambda block: block.Label.min()).to_list()
        # Encode the label into one-hot if necessary.
        if one_hot_label:
            y = [Dataset.to_onehot(label) for label in y]

        # Convert the dataset into numpy arrays for training/inference.
        return pd.concat(frames, axis=1).fillna(fillna_value).to_numpy(), np.array(y)

    @staticmethod
    def to_onehot(label: int) -> List[int]:
        return [(1 if i == label else 0) for i in range(5)]

    @staticmethod
    def from_onehot(label: List[int], approximate=False) -> int:
        if not approximate:
            return label.index(1)
        else:
            return np.argmax(label)
