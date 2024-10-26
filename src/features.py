from dataclasses import dataclass
from typing import Callable
from pandas import DataFrame


@dataclass
class Feature:
    name: str
    func: Callable[[DataFrame], float]


# Features we want to use for training/inference.
syncan_features = (
    [
        Feature(
            name=f"signal{sig}_stdev",
            func=lambda x: (
                x[f"Signal{sig}_of_ID"].std()
                if not (x[f"Signal{sig}_of_ID"].isnull().all())
                else 0.0
            ),
        )
        for sig in range(1, 5)
    ]
    + [
        Feature(
            name=f"signal{sig}_avg",
            func=lambda x: (
                x[f"Signal{sig}_of_ID"].mean()
                if not (x[f"Signal{sig}_of_ID"].isnull().all())
                else 0.0
            ),
        )
        for sig in range(1, 5)
    ]
    + [
        Feature(
            name=f"signal{sig}_median",
            func=lambda x: (
                x[f"Signal{sig}_of_ID"].median()
                if not (x[f"Signal{sig}_of_ID"].isnull().all())
                else 0.0
            ),
        )
        for sig in range(1, 5)
    ]
    + [Feature(name="message_count", func=lambda x: len(x.index))]
)

ch_ids_features = (
    [
        Feature(
            name=f"signal{sig}_stdev",
            func=lambda x: (
                x[f"DATA{sig}"].std() if not (x[f"DATA{sig}"].isnull().all()) else 0.0
            ),
        )
        for sig in range(0, 8)
    ]
    + [
        Feature(
            name=f"signal{sig}_avg",
            func=lambda x: (
                x[f"DATA{sig}"].mean() if not (x[f"DATA{sig}"].isnull().all()) else 0.0
            ),
        )
        for sig in range(0, 8)
    ]
    + [
        Feature(
            name=f"signal{sig}_median",
            func=lambda x: (
                x[f"DATA{sig}"].median()
                if not (x[f"DATA{sig}"].isnull().all())
                else 0.0
            ),
        )
        for sig in range(0, 8)
    ]
    + [Feature(name="message_count", func=lambda x: len(x.index))]
)
