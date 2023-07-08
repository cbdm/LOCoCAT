import pandas as pd
import numpy as np
from os.path import join, pardir
from random import seed

seed(0)

EXPORT_PATH = join(pardir, "data")

og_path = join(EXPORT_PATH, "syncan_attack_only.csv")

df = pd.read_csv(og_path)

block_starts = (
    df.Time.diff().fillna(500)
    // 500
)

block_starts = block_starts[block_starts > 0]
blocks = (
    pd.DataFrame(block_starts)
    .reset_index()
    .reset_index()
    .drop(columns="Time")
    .rename(columns={"index": "Index", "level_0": "Block"})
)

df = pd.merge_asof(df, blocks, on="Index")

block_nums = np.arange(df.Block.max() + 1)
block_pos = block_nums.copy()
np.random.shuffle(block_pos)

pos_df = pd.DataFrame({"Block": block_nums, "Pos": block_pos})

df = pd.merge_asof(df, pos_df, on="Block").rename(columns={"Index": "ogIndex"})
df = df.sort_values(by=["Pos", "ogIndex"])

df = df.reset_index(drop=True)
df[['Label', 'Time', 'ID', 'Signal1_of_ID', 'Signal2_of_ID', 'Signal3_of_ID', 'Signal4_of_ID', 'ogIndex', 'Block', 'Pos']].to_csv(join(EXPORT_PATH, "syncan_shuffled_blocks.csv"), index_label="Index")
