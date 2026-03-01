import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

DATA_DIR = "data"
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

def main():
    # List .mat files you currently have
    mats = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".mat")])
    if not mats:
        raise FileNotFoundError(f"No .mat files found in {DATA_DIR}/")

    print("Found .mat files:")
    for f in mats:
        print(" -", f)

    # Load the first one for now
    file_path = os.path.join(DATA_DIR, mats[0])
    print("\nLoading:", file_path)

    mat = load_mat_any(file_path)
    print("Keys in .mat:", list(mat.keys()))

    eeg_key, eeg = pick_eeg_matrix(mat, expected_channels=19)
    print("\nChosen EEG variable:", eeg_key)
    print("EEG shape (channels x samples):", eeg.shape)

    duration_sec = eeg.shape[1] / SFREQ
    print(f"Approx duration: {duration_sec/60:.2f} minutes ({duration_sec:.1f} seconds)")

    plot_10_seconds(eeg, channel_index=0, title=mats[0])

if __name__ == "__main__":
    main()