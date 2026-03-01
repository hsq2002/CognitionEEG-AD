import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

DATA_DIR = "patient_data"
SFREQ = 500  # from dataset overview

def load_mat_any(path):
    """Load .mat (non-v7.3). Return dict without __ keys."""
    mat = sio.loadmat(path)
    return {k: v for k, v in mat.items() if not k.startswith("__")}

def pick_eeg_matrix(mat_dict, expected_channels=19):
    """
    Find a 2D numeric matrix that looks like EEG.
    Prefer one with a 19 in one dimension.
    Returns (key, eeg_array in channels x samples).
    """
    candidates = []
    for k, v in mat_dict.items():
        if isinstance(v, np.ndarray) and v.ndim == 2 and np.issubdtype(v.dtype, np.number):
            candidates.append((k, v))

    if not candidates:
        raise ValueError("No 2D numeric arrays found in the .mat file.")

    # Prefer candidate with 19 channels
    best_k, best_v = None, None
    for k, v in candidates:
        if expected_channels in v.shape:
            best_k, best_v = k, v
            break

    # Otherwise just take the largest 2D numeric array
    if best_v is None:
        best_k, best_v = max(candidates, key=lambda kv: kv[1].size)

    eeg = np.array(best_v, dtype=float)

    # Ensure channels x samples
    if eeg.shape[0] != expected_channels and eeg.shape[1] == expected_channels:
        eeg = eeg.T

    return best_k, eeg

def plot_10_seconds(eeg, channel_index=0, title="EEG"):
    samples = int(10 * SFREQ)
    if eeg.shape[1] < samples:
        raise ValueError(f"Not enough samples to plot 10 seconds. Have {eeg.shape[1]} samples.")

    t = np.arange(samples) / SFREQ
    y = eeg[channel_index, :samples]

    plt.figure(figsize=(12, 4))
    plt.plot(t, y)
    plt.title(title + f" | channel index {channel_index}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (dataset units)")
    plt.tight_layout()
    plt.show()

import pandas as pd

def main():
    # 1) Load participants file
    participants_path = os.path.join(DATA_DIR, "participants.tsv")
    participants = pd.read_csv(participants_path, sep="\t")

    # 2) Pick 1 AD (Group == 'A') and 1 CN/HC (Group == 'C')
    ad_id = participants.loc[participants["Group"] == "A", "participant_id"].iloc[0]
    cn_id = participants.loc[participants["Group"] == "C", "participant_id"].iloc[0]

    print("Chosen AD subject:", ad_id)
    print("Chosen CN subject:", cn_id)

    # 3) Build file paths (matches your naming pattern)
    ad_file = os.path.join(DATA_DIR, f"{ad_id}_task-eyesclosed.mat")
    cn_file = os.path.join(DATA_DIR, f"{cn_id}_task-eyesclosed.mat")

    # 4) Load both EEG matrices
    ad_mat = load_mat_any(ad_file)
    cn_mat = load_mat_any(cn_file)

    ad_key, ad_eeg = pick_eeg_matrix(ad_mat, expected_channels=19)
    cn_key, cn_eeg = pick_eeg_matrix(cn_mat, expected_channels=19)

    print(f"\nAD EEG var: {ad_key} | shape={ad_eeg.shape}")
    print(f"CN EEG var: {cn_key} | shape={cn_eeg.shape}")

    # 5) Report recording durations
    ad_duration_sec = ad_eeg.shape[1] / SFREQ
    cn_duration_sec = cn_eeg.shape[1] / SFREQ
    print(f"AD duration: {ad_duration_sec/60:.2f} min ({ad_duration_sec:.1f} sec)")
    print(f"CN duration: {cn_duration_sec/60:.2f} min ({cn_duration_sec:.1f} sec)")

    # 6) Plot 10 seconds from the SAME channel for both
    channel_index = 0  # change if you want a different channel
    samples = int(10 * SFREQ)
    t = np.arange(samples) / SFREQ

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, cn_eeg[channel_index, :samples])
    plt.title(f"CN ({cn_id}) - 10 sec | channel index {channel_index}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.plot(t, ad_eeg[channel_index, :samples])
    plt.title(f"AD ({ad_id}) - 10 sec | channel index {channel_index}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()