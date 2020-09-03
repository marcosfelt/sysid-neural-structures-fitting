import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
import pickle

__all__ = [
    "INPUT_COLUMNS",
    "INPUT_COLUMNS_REPEAT",
    "OUTPUT_COLUMNS",
    "OUTPUT_COLUMNS_REPEAT",
    "dataload",
]

INPUT_COLUMNS = [
    "reboiler_feed_ratio",
    "reflux_feed_ratio",
    "distllate_feed_ratio",
    "ethane_feed_composition",
    "top_pressure",
]
INPUT_COLUMNS_REPEAT = [
    f"{col}_{noise_level}"
    for noise_level in ["n0", "n10", "n20", "n30"]
    for col in INPUT_COLUMNS
]
OUTPUT_COLUMNS = [
    "top_ethane_composition",
    "bottom_ethylene_composition",
    "top_bottom_differential_pressure",
]
OUTPUT_COLUMNS_REPEAT = [
    f"{col}_{noise_level}"
    for noise_level in ["n0", "n10", "n20", "n30"]
    for col in OUTPUT_COLUMNS
]


def dataload(
    data_file: str = "data/distill.dat",
    input_columns: list = INPUT_COLUMNS_REPEAT,
    output_columns: list = OUTPUT_COLUMNS_REPEAT,
    ts: float = 15.0,
    valid_pct: float = 0.2,
    standardize_inputs: bool = True,
    standardize_outputs: bool = True,
) -> pd.DataFrame:
    """Load data and split into training and validation set.

    Parameters
    ----------
    data_file: str
        Name of the file. Defaults to data/distill2.dat
    columns : list
        Names of the columns in the dataset
    index_column : int
        Index column of the data file. Defaults to 0 but could be None if no index.
    ts : float
        The time elapsed between each sample. Defaults to 15 minutes.
    valid_pct : float
        Fraction of dataset used for validation. Defaults to 0.2. Dataset is
        split by time instead of randomly.
    standardize_outputs : bool
        Standardize variable in columns. Defaults to True.

    Returns
    -------
    df, train_U, train_Y, valid_U, valid_Y

    """
    # Read in datafile
    df = pd.read_csv(
        data_file,
        names=input_columns + output_columns,
        header=None,
        delim_whitespace=True,
    )

    # Sampling times
    df["time"] = ts * df.index.values
    df.set_index("time", inplace=True)

    # Train-validation split
    n_data = df.shape[0]
    n_valid = int(valid_pct * n_data)
    n_train = n_data - n_valid
    df_train = df.iloc[0:n_train]
    df_valid = df.iloc[n_train:]

    train_U = df_train[input_columns]
    train_Y = df_train[output_columns]
    valid_U = df_valid[input_columns]
    valid_Y = df_valid[output_columns]

    # Scaling
    if standardize_inputs:
        mean = train_U.mean()
        std = train_U.std()
        train_U = (train_U - mean) / std
        valid_U = (valid_U - mean) / std

    if standardize_outputs:
        mean = train_Y.mean()
        std = train_Y.std()
        train_Y = (train_Y - mean) / std
        valid_Y = (valid_Y - mean) / std

    return df, train_U, train_Y, valid_U, valid_Y


if __name__ == "__main__":
    dataload()
