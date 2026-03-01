import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

DATA_DIR = "data"
SFREQ = 500  # from dataset overview

def load_eeg_mat(filepath):
    mat = sio.loadmat(filepath)
    keys = [k for k in mat.keys() if not k.startswith("__")]

    for k in keys:
        if isinstance(mat[k], np.ndarray) and mat[k].ndim == 2:
            data = mat[k]
            break

    # Ensure shape is Channels x Samples
    if data.shape[0] != 19:
        data = data.T

    return data

def main():
    participants = pd.read_csv(os.path.join(DATA_DIR, "participants.tsv"), sep="\t")

    # Adjust column name if needed (group, diagnosis, etc.)
    cn_id = participants[participants["group"].isin(["CN", "HC"])]["participant_id"].iloc[0]
    ad_id = participants[participants["group"] == "AD"]["participant_id"].iloc[0]

    print("CN subject:", cn_id)
    print("AD subject:", ad_id)

    cn_file = os.path.join(DATA_DIR, f"{cn_id}_task-eyesclosed.mat")
    ad_file = os.path.join(DATA_DIR, f"{ad_id}_task-eyesclosed.mat")

    cn_eeg = load_eeg_mat(cn_file)
    ad_eeg = load_eeg_mat(ad_file)

    print("EEG shape:", cn_eeg.shape)

    # Plot 10 seconds
    samples = 10 * SFREQ
    channel_index = 0  # change later if needed

    t = np.arange(samples) / SFREQ

    plt.figure(figsize=(12,5))

    plt.subplot(2,1,1)
    plt.plot(t, cn_eeg[channel_index, :samples])
    plt.title("CN - 10 sec EEG")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.subplot(2,1,2)
    plt.plot(t, ad_eeg[channel_index, :samples])
    plt.title("AD - 10 sec EEG")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()