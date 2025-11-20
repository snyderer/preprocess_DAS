from pathlib import Path
import h5py
import pandas as pd
import numpy as np

def fix_timestamp_in_chunk(filepath):
    with h5py.File(filepath, 'r+') as f:  # r+ = read/write without truncating the file
        if 'timestamp' not in f:
            print(f"[SKIP] No timestamp dataset in {filepath}")
            return

        # Read existing stored timestamp (varies in dtype)
        ts_raw = f['timestamp'][()]

        # Convert depending on dtype
        try:
            # If already float-like, interpret as naive seconds → convert to pandas, treat as local
            if np.isscalar(ts_raw) and isinstance(ts_raw, (float, np.floating, int, np.integer)):
                ts_corrected = pd.Timestamp(ts_raw, unit='s').timestamp()

            # If numpy.datetime64
            elif isinstance(ts_raw, np.datetime64):
                ts_corrected = pd.Timestamp(ts_raw).timestamp()

            # If it came back as bytes (string representation)
            elif isinstance(ts_raw, (bytes, str)):
                ts_corrected = pd.Timestamp(ts_raw).timestamp()

            else:
                raise ValueError(f"Unknown timestamp type: {type(ts_raw)}")

        except Exception as e:
            print(f"[ERROR] Failed to parse timestamp in {filepath}: {e}")
            return

        # Overwrite dataset with corrected float
        del f['timestamp']  # delete old dataset
        f.create_dataset('timestamp', data=ts_corrected)
        print(f"[FIXED] {filepath.name}: {ts_raw} → {ts_corrected} ({pd.Timestamp.utcfromtimestamp(ts_corrected)})")


# === Example: process all chunk files in a directory ===
def fix_all_timestamps_in_dir(directory):
    chunk_files = list(Path(directory).glob("*.h5"))
    # optionally skip settings.h5
    chunk_files = [f for f in chunk_files if f.name != "settings.h5"]

    for file in chunk_files:
        fix_timestamp_in_chunk(file)

# Usage
fix_all_timestamps_in_dir(r"F:\ooi_optasense_north_c2_full")