"""Shared dataset loading for the BME688 master CSVs."""

import glob
import os

import pandas as pd

COLS = [
    "Label", "Timestamp", "Temp_C", "Humidity_Pct", "Pressure_hPa",
    "Gas_Step_0(kOhm)", "Gas_Step_1(kOhm)", "Gas_Step_2(kOhm)",
    "Gas_Step_3(kOhm)", "Gas_Step_4(kOhm)", "Gas_Step_5(kOhm)",
    "Gas_Step_6(kOhm)", "Gas_Step_7(kOhm)", "Gas_Step_8(kOhm)",
    "Gas_Step_9(kOhm)",
]


def load_dataset(data_dir):
    csv_files = glob.glob(os.path.join(data_dir, "dataset_*_master.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No dataset_*_master.csv files found in {data_dir}")

    frames = []
    for f in sorted(csv_files):
        temp_df = pd.read_csv(f, header=None)
        if str(temp_df.iloc[0, 0]) == "Label":
            temp_df = temp_df.iloc[1:].reset_index(drop=True)
        temp_df.columns = COLS
        for c in COLS[2:]:
            temp_df[c] = pd.to_numeric(temp_df[c], errors="coerce")
        frames.append(temp_df.dropna())

    return pd.concat(frames, ignore_index=True)
