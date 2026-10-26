"""Build the clean, labelled dataset in data/ from the raw Sensor Logger exports in raw/.

Each raw recording folder contains Accelerometer.csv and Gyroscope.csv (iPhone X,
100 Hz, gravity removed). This script:
  1. merges the two sensors on their shared nanosecond timestamp,
  2. trims 0.5 s from each end (phone tap to start/stop the app),
  3. resamples onto an exact 100 Hz grid (linear interpolation) to harmonise sampling,
  4. writes data/train/<activity>/<activity>_NN.csv and data/test/*.csv.

Test files (never used for training):
  - the three continuous "longer_videos" sessions (standing -> walking -> jumping -> still),
    labelled per sample using segment boundaries read from the signal + video
  - the standing clip recorded on 2026-10-23 (different day from the training session)
"""
import glob
import os
import re

import numpy as np
import pandas as pd

RAW = os.path.join("raw", "data collected")
OUT = "data"
FS = 100  # target sampling rate (Hz)
TRIM = 0.5  # seconds trimmed from each end

# Segment boundaries (seconds from start of recording) for the continuous test sessions.
LONG_SEGMENTS = {
    "longer_videos_1": [("standing", 0, 9.0), ("walking", 9.0, 18.5), ("jumping", 18.5, 30.0), ("still", 30.0, 1e9)],
    "longer_videos_2": [("standing", 0, 10.5), ("walking", 10.5, 21.5), ("jumping", 21.5, 34.5), ("still", 34.5, 1e9)],
    "longer_videos_3": [("standing", 0, 11.0), ("walking", 11.0, 21.0), ("jumping", 21.0, 33.5), ("still", 33.5, 1e9)],
}


def load_recording(folder):
    """Merge accelerometer + gyroscope, trim edges, resample to an exact FS grid."""
    acc = pd.read_csv(os.path.join(folder, "Accelerometer.csv"))
    gyr = pd.read_csv(os.path.join(folder, "Gyroscope.csv"))
    acc = acc.rename(columns={c: f"acc_{c}" for c in "xyz"})
    gyr = gyr.rename(columns={c: f"gyro_{c}" for c in "xyz"})
    df = pd.merge(acc, gyr.drop(columns="seconds_elapsed"), on="time", how="inner")
    df = df.sort_values("time").drop_duplicates("time")

    t = (df["time"].values - df["time"].values[0]) / 1e9
    grid = np.arange(TRIM, t[-1] - TRIM, 1 / FS)
    out = pd.DataFrame({"timestamp": df["time"].values[0] + (grid * 1e9).astype("int64"),
                        "seconds_elapsed": np.round(grid, 3)})
    for col in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
        out[col] = np.interp(grid, t, df[col].values)
    return out


def main():
    folders = sorted(f for f in glob.glob(os.path.join(RAW, "*")) if os.path.isdir(f))
    counters = {}
    for f in folders:
        name = os.path.basename(f)
        df = load_recording(f)
        long_match = re.match(r"(longer_videos_\d)", name)
        if long_match:
            key = long_match.group(1)
            df["label"] = ""
            for act, a, b in LONG_SEGMENTS[key]:
                df.loc[(df.seconds_elapsed >= a) & (df.seconds_elapsed < b), "label"] = act
            path = os.path.join(OUT, "test", f"test_session_{key[-1]}_continuous.csv")
        else:
            act = re.match(r"video_([a-z]+)_\d+", name).group(1)
            df["label"] = act
            if "2026-10-23" in name:
                path = os.path.join(OUT, "test", f"test_{act}_different_day.csv")
            else:
                counters[act] = counters.get(act, 0) + 1
                path = os.path.join(OUT, "train", act, f"{act}_{counters[act]:02d}.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"{name:45s} -> {path}  ({len(df)} samples, {len(df) / FS:.1f}s)")


if __name__ == "__main__":
    main()
