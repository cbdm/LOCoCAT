from dataclasses import dataclass
from typing import Callable
from pandas import DataFrame


@dataclass
class Filter:
    name: str
    func: Callable[[DataFrame], DataFrame]


# Filters for the feature set.
## Filter by time.
start_50ms = Filter(name="start50ms", func=lambda x: x[x.Time <= (50 + x.Time.min())])
start_100ms = Filter(
    name="start100ms", func=lambda x: x[x.Time <= (100 + x.Time.min())]
)
start_200ms = Filter(
    name="start200ms", func=lambda x: x[x.Time <= (200 + x.Time.min())]
)
start_500ms = Filter(
    name="start500ms", func=lambda x: x[x.Time <= (500 + x.Time.min())]
)
start_1000ms = Filter(
    name="start1000ms", func=lambda x: x[x.Time <= (1000 + x.Time.min())]
)

# Data should be already pre-processed by the prepare_data script.
# If you want to apply any extra filters/transforms to it, you can populate the list below.
preprocessing_filters = []

# Train models using different sized windows of the beginning of the attack.
training_filters = [
    [],
    [start_1000ms],
    [start_500ms],
    [start_200ms],
    [start_100ms],
    [start_50ms],
]
