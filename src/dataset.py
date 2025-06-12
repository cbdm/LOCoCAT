from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from features import Feature
from filters import Filter


class Dataset(object):
    def __init__(
        self,
        *,
        name: str,
        path: str,
        filters: List[Filter] = [],
        attack_split_factor: int = 0,
        pre_labeled_blocks_column: str = "",
        prefill_na_value: Optional[float] = None,
    ):
        self.name = name
        self._path = path
        self._dataset_filters = filters
        assert bool(attack_split_factor) ^ bool(
            pre_labeled_blocks_column
        ), "You need to provide exactly one of: (i) a column name that contains block labels, (ii) an attack split factor to calculate blocks."
        self._attack_split_factor = attack_split_factor
        self._pre_labeled_blocks_column = pre_labeled_blocks_column
        self._prefill_na_value = prefill_na_value
        self._data = None

    def _load(self, quiet: bool) -> None:
        df = pd.read_csv(self._path)

        if not quiet:
            print(f"Loaded dataset from '{self._path}'")

        # Check if we should already replace NaN values with a provided one.
        if self._prefill_na_value is not None:
            df = df.fillna(self._prefill_na_value)

        # Filter the dataset if necessary.
        for f in self._dataset_filters:
            if not quiet:
                print(f"Filtering dataset with with '{f.name=}'")
            df = f.func(df)

        # Check if the blocks are already labeled in the dataset.
        # You can create pre-labeled files with the prepare_data script and the --blocks option.
        if self._pre_labeled_blocks_column:
            if not quiet:
                print(
                    f"Using column '{self._pre_labeled_blocks_column}' to group messages in blocks."
                )

            # Attack blocks are already labeled, do not need to recalculate them.
            # So we simply rename the arbitrary column name given by the user, group it, and return.
            self._data = df.rename(
                columns={self._pre_labeled_blocks_column: "Block"}
            ).groupby("Block")
            return

        if not quiet:
            print(
                f"Using '{self._attack_split_factor}' as split factor to group messages in blocks."
            )

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

    def get_data(self, quiet: bool) -> pd.DataFrame:
        if self._data is None:
            if not quiet:
                print("Loading data for the first time.")
            self._load(quiet)
        return self._data

    def create_feature_set(
        self,
        *,
        filters: List[Filter] = [],
        features: List[Feature] = [],
        one_hot_label: bool = False,
        fillna_value: float = 0.0,
        remove_classes: Set[int] = set(),
        quiet: bool = False,
    ) -> Tuple[np.ndarray, np.array]:
        df = self.get_data(quiet)
        frames = []

        # If needed, remove datapoints for the given classes.
        if remove_classes is not None:
            df = (
                df.filter(lambda block: block["Label"].min() not in remove_classes)
                .reset_index(level=0, drop=True)
                .groupby("Block")
            )

        # Filter the dataset if necessary.
        for i, f in enumerate(filters):
            if not quiet:
                print(f"Filtering with '{f.name=}' ({i+1} / {len(filters)})")
            df = df.apply(f.func).reset_index(level=0, drop=True).groupby("Block")

        # Calculate the features.
        for i, feat in enumerate(features):
            if not quiet:
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
