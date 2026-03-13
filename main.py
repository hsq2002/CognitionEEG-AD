import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import welch
import pandas as pd
from scipy import stats

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "patient_data")
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

def compute_band_powers(eeg, sfreq=500, nperseg=2048):
    """
    Compute Power Spectral Density (PSD) using Welch method and extract band powers.
    
    Parameters:
    -----------
    eeg : ndarray of shape (channels, samples)
        EEG signal
    sfreq : int
        Sampling frequency in Hz
    nperseg : int
        Length of each segment for Welch method
    
    Returns:
    --------
    dict : Contains 'freqs', 'psd' (channels x freqs), and band powers
    """
    n_channels = eeg.shape[0]
    
    # Compute PSD using Welch method
    freqs, psd = welch(eeg, sfreq, nperseg=nperseg, axis=1)
    
    # Define frequency bands
    bands = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30)
    }
    
    # Extract band powers (average power in each band)
    band_powers = {}
    for band_name, (low_freq, high_freq) in bands.items():
        band_mask = (freqs >= low_freq) & (freqs < high_freq)
        # Average power across frequencies in the band for each channel
        band_powers[band_name] = np.mean(psd[:, band_mask], axis=1)
    
    return {
        'freqs': freqs,
        'psd': psd,
        'bands': bands,
        'band_powers': band_powers
    }

def create_band_power_table(participant_id, band_powers_dict, n_channels=19):
    """
    Create a pandas DataFrame with band powers for all channels of a participant.
    
    Parameters:
    -----------
    participant_id : str
        Subject identifier
    band_powers_dict : dict
        Output from compute_band_powers()
    n_channels : int
        Number of EEG channels
    
    Returns:
    --------
    DataFrame with columns: participant_id, Channel, Delta, Theta, Alpha, Beta
    """
    band_powers = band_powers_dict['band_powers']
    
    data = []
    for ch_idx in range(n_channels):
        row = {
            'Participant_ID': participant_id,
            'Channel': f'Ch{ch_idx}',
            'Delta': band_powers['Delta'][ch_idx],
            'Theta': band_powers['Theta'][ch_idx],
            'Alpha': band_powers['Alpha'][ch_idx],
            'Beta': band_powers['Beta'][ch_idx]
        }
        data.append(row)
    
    return pd.DataFrame(data)

def process_all_subjects(participants_df, data_dir, sfreq=500):
    """
    Process all subjects and compute average band powers across channels.
    
    Returns:
    --------
    DataFrame with columns: participant_id, Group, Delta, Theta, Alpha, Beta
    """
    results = []
    
    for idx, row in participants_df.iterrows():
        participant_id = row['participant_id']
        group = row['Group']
        
        file_path = os.path.join(data_dir, f"{participant_id}_task-eyesclosed.mat")
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found for {participant_id}, skipping.")
            continue
        
        try:
            # Load EEG data
            mat_dict = load_mat_any(file_path)
            _, eeg = pick_eeg_matrix(mat_dict, expected_channels=19)
            
            # Compute band powers
            analysis = compute_band_powers(eeg, sfreq=sfreq)
            band_powers = analysis['band_powers']
            
            # Average across channels for each band
            avg_bands = {}
            for band in ['Delta', 'Theta', 'Alpha', 'Beta']:
                avg_bands[band] = np.mean(band_powers[band])
            
            # Add to results
            result = {
                'participant_id': participant_id,
                'Group': group,
                **avg_bands
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {participant_id}: {e}")
            continue
    
    return pd.DataFrame(results)

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

    # 7) Frequency Analysis - Compute PSD and Band Powers
    print("\n" + "="*80)
    print("FREQUENCY ANALYSIS - BAND POWER COMPUTATION")
    print("="*80)

    # Compute band powers for both subjects
    ad_analysis = compute_band_powers(ad_eeg, sfreq=SFREQ)
    cn_analysis = compute_band_powers(cn_eeg, sfreq=SFREQ)

    # Create tables
    ad_table = create_band_power_table(ad_id, ad_analysis)
    cn_table = create_band_power_table(cn_id, cn_analysis)

    print(f"\n--- {ad_id} (Alzheimer's Disease) - Band Power per Channel ---")
    print(ad_table.to_string(index=False))

    print(f"\n--- {cn_id} (Control) - Band Power per Channel ---")
    print(cn_table.to_string(index=False))

    # 8) Comparison - Average band powers across channels
    print("\n" + "="*80)
    print("COMPARISON: Average Band Power (Averaged across all channels)")
    print("="*80)

    comparison_data = []
    for band in ['Delta', 'Theta', 'Alpha', 'Beta']:
        ad_avg = ad_table[band].mean()
        cn_avg = cn_table[band].mean()
        diff = ad_avg - cn_avg
        pct_diff = (diff / cn_avg * 100) if cn_avg != 0 else 0
        
        comparison_data.append({
            'Band': band,
            f'{ad_id} (AD)': ad_avg,
            f'{cn_id} (CN)': cn_avg,
            'Difference': diff,
            'Percent_Diff_%': pct_diff
        })

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    # 9) Visualize PSD comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(ad_analysis['freqs'], ad_analysis['psd'][channel_index], label='AD', linewidth=2)
    plt.plot(cn_analysis['freqs'], cn_analysis['psd'][channel_index], label='CN', linewidth=2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (V²/Hz)")
    plt.title(f"PSD Comparison - Channel {channel_index}")
    plt.legend()
    plt.xlim([0, 40])
    plt.grid(True, alpha=0.3)

    # Plot band powers as bar chart
    plt.subplot(1, 2, 2)
    bands_list = ['Delta', 'Theta', 'Alpha', 'Beta']
    ad_means = [ad_table[band].mean() for band in bands_list]
    cn_means = [cn_table[band].mean() for band in bands_list]

    x = np.arange(len(bands_list))
    width = 0.35

    plt.bar(x - width/2, ad_means, width, label='AD', alpha=0.8)
    plt.bar(x + width/2, cn_means, width, label='CN', alpha=0.8)

    plt.xlabel("Frequency Band")
    plt.ylabel("Average Power (V²/Hz)")
    plt.title("Average Band Power Comparison")
    plt.xticks(x, bands_list)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    # TASK 3: Statistical Comparison Between Groups
    print("\n" + "="*80)
    print("TASK 3: Statistical Comparison Between Groups")
    print("="*80)
    
    print(f"Loaded {len(participants)} participants")
    print(f"AD subjects: {len(participants[participants['Group'] == 'A'])}")
    print(f"Control subjects: {len(participants[participants['Group'] == 'C'])}")
    
    # Process all subjects
    print("\nProcessing all subjects...")
    band_power_df = process_all_subjects(participants, DATA_DIR, sfreq=SFREQ)
    
    print(f"Successfully processed {len(band_power_df)} subjects")
    print(f"AD: {len(band_power_df[band_power_df['Group'] == 'A'])}")
    print(f"Control: {len(band_power_df[band_power_df['Group'] == 'C'])}")
    
    # Create boxplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Alpha power boxplot
    alpha_data = [band_power_df[band_power_df['Group'] == 'A']['Alpha'], 
                  band_power_df[band_power_df['Group'] == 'C']['Alpha']]
    ax1.boxplot(alpha_data, tick_labels=['AD', 'Control'])
    ax1.set_title('Alpha Power Distribution')
    ax1.set_ylabel('Power (V²/Hz)')
    ax1.grid(True, alpha=0.3)
    
    # Theta power boxplot
    theta_data = [band_power_df[band_power_df['Group'] == 'A']['Theta'], 
                  band_power_df[band_power_df['Group'] == 'C']['Theta']]
    ax2.boxplot(theta_data, tick_labels=['AD', 'Control'])
    ax2.set_title('Theta Power Distribution')
    ax2.set_ylabel('Power (V²/Hz)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical testing
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    # Separate groups
    ad_alpha = band_power_df[band_power_df['Group'] == 'A']['Alpha']
    cn_alpha = band_power_df[band_power_df['Group'] == 'C']['Alpha']
    ad_theta = band_power_df[band_power_df['Group'] == 'A']['Theta']
    cn_theta = band_power_df[band_power_df['Group'] == 'C']['Theta']
    
    # Alpha power comparison
    print("\n--- Alpha Power ---")
    alpha_ad_mean = ad_alpha.mean()
    alpha_cn_mean = cn_alpha.mean()
    alpha_diff = alpha_ad_mean - alpha_cn_mean
    print(".6f")
    print(".6f")
    print(".6f")
    
    # t-test for Alpha
    t_stat_alpha, p_val_alpha = stats.ttest_ind(ad_alpha, cn_alpha)
    print(".4f")
    print(".4e")
    
    # Theta power comparison
    print("\n--- Theta Power ---")
    theta_ad_mean = ad_theta.mean()
    theta_cn_mean = cn_theta.mean()
    theta_diff = theta_ad_mean - theta_cn_mean
    print(".6f")
    print(".6f")
    print(".6f")
    
    # t-test for Theta
    t_stat_theta, p_val_theta = stats.ttest_ind(ad_theta, cn_theta)
    print(".4f")
    print(".4e")
    
    # Observations
    print("\n" + "="*80)
    print("OBSERVATIONS")
    print("="*80)
    
    print("Alpha Power:")
    if p_val_alpha < 0.05:
        print(".4e")
        if alpha_diff > 0:
            print("  - AD subjects show higher Alpha power than Controls")
        else:
            print("  - AD subjects show lower Alpha power than Controls")
    else:
        print(".4f")
    
    print("\nTheta Power:")
    if p_val_theta < 0.05:
        print(".4e")
        if theta_diff > 0:
            print("  - AD subjects show higher Theta power than Controls")
        else:
            print("  - AD subjects show lower Theta power than Controls")
    else:
        print(".4f")
    
    print("\nAnalysis Complete!")

if __name__ == "__main__":
    main()