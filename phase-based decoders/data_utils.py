import scipy.io
import numpy as np
import scipy.signal as signal
from scipy.signal import butter, filtfilt
import torch
import os

# Import from config
from config import (
    LFP_LOWCUT, LFP_HIGHCUT, LFP_FS, LFP_FILTER_ORDER,
    LFP_SEGMENT_START_MS, LFP_SEGMENT_END_MS,
    SPIKE_SEGMENT_START_MS, SPIKE_SEGMENT_END_MS,
    N_LFP_CHANNELS, N_SPIKE_CHANNELS, N_BRANCHES,
    FEF_LFP_CHANNEL_ID, CONDITION_SOURCE_CHANNEL_ID
)

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Designs a Butterworth bandpass filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Applies a Butterworth bandpass filter to data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=-1)  # Apply filter along the time axis
    return y


def zscore_normalize(data):
    """
    Performs Z-score normalization along the last axis.
    """
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    # Add a small epsilon to std to prevent division by zero if std is 0
    normalized_data = (data - mean) / (std + 1e-8)
    return normalized_data


def load_and_preprocess_data(probe_mapping,
                             lfp_path_template, spike_path_template):
    """
    Loads and preprocesses LFP, spike, and condition data from relative paths.

    Args:
        probe_mapping (dict): Mapping from condition labels to coordinates.
        lfp_path_template (str): String template for LFP file paths.
        spike_path_template (str): String template for Spike file paths.

    Returns:
        tuple: (X, Y_coords, Y_original_labels)
               X: np.ndarray of shape (n_samples, n_branches, 2, time_length)
               Y_coords: np.ndarray of shape (n_samples, 2) representing spatial coordinates
               Y_original_labels: np.ndarray of shape (n_samples,) for stratification
    """
    print("Loading and preprocessing data from 'sample_data' directory...")

    v4_X_list = []
    spk_X_list = []

    # Load and preprocess V4 LFP data (Channels 1-16)
    print(f"Loading and preprocessing {N_LFP_CHANNELS} V4 LFP channels...")
    for cid in range(1, N_LFP_CHANNELS + 1):
        file_path = lfp_path_template.format(channel_id=cid)
        dd = scipy.io.loadmat(file_path)
        # Assuming the key inside the .mat file is 'ProbeLFP_fix'
        lfp_full_length = np.array(dd['ProbeLFP_fix'])

        lfp_filtered_full = butter_bandpass_filter(lfp_full_length, LFP_LOWCUT, LFP_HIGHCUT, LFP_FS, LFP_FILTER_ORDER)
        v4_X_list.append(lfp_filtered_full[:, LFP_SEGMENT_START_MS:LFP_SEGMENT_END_MS])

    # Load and preprocess V4 Spike data (Channels 1-16)
    print(f"Loading and preprocessing {N_SPIKE_CHANNELS} V4 Spike channels...")
    for cid in range(1, N_SPIKE_CHANNELS + 1):
        file_path = spike_path_template.format(channel_id=cid)
        dd = scipy.io.loadmat(file_path)
        # Assuming the key inside the .mat file is 'SpikeProbe_fix'
        spike_full_length = np.array(dd['SpikeProbe_fix'])
        spk_X_list.append(spike_full_length[:, SPIKE_SEGMENT_START_MS:SPIKE_SEGMENT_END_MS])

    # Load and preprocess FEF LFP data (from channel 17)
    print(f"Loading and preprocessing FEF LFP data (channel {FEF_LFP_CHANNEL_ID})...")
    fef_lfp_file_path = lfp_path_template.format(channel_id=FEF_LFP_CHANNEL_ID)
    dd_fef = scipy.io.loadmat(fef_lfp_file_path)
    fef_lfp_full_length = np.array(dd_fef['ProbeLFP_fix'])

    fef_lfp_filtered_full = butter_bandpass_filter(fef_lfp_full_length, LFP_LOWCUT, LFP_HIGHCUT, LFP_FS, LFP_FILTER_ORDER)
    fef_X = fef_lfp_filtered_full[:, LFP_SEGMENT_START_MS:LFP_SEGMENT_END_MS]

    # Load condition (label) information (from channel 1)
    print(f"Loading condition data (from LFP channel {CONDITION_SOURCE_CHANNEL_ID})...")
    conditions_file_path = lfp_path_template.format(channel_id=CONDITION_SOURCE_CHANNEL_ID)
    dd_cond = scipy.io.loadmat(conditions_file_path)
    # Assuming the key inside the .mat file is 'Conditions_fix'
    Y_original_labels = np.array(dd_cond['Conditions_fix'])
    Y_original_labels = np.squeeze(Y_original_labels)
    Y_coords = np.array([probe_mapping[int(val)] for val in Y_original_labels])

    # Hilbert Transform and phase extraction
    print("Calculating Hilbert transform and phase differences for LFP data...")
    v4_phase_list = []
    for i in range(N_LFP_CHANNELS):
        sig = signal.hilbert(v4_X_list[i], axis=1)
        v4_phase_list.append(np.angle(sig))

    fef_signal = signal.hilbert(fef_X, axis=1)
    fef_phase = np.angle(fef_signal)

    delta_phase_list = []
    for i in range(N_LFP_CHANNELS):
        dp = (v4_phase_list[i] - fef_phase + np.pi) % (2 * np.pi) - np.pi
        delta_phase_list.append(dp)

    # Concatenate (LFP delta_phase, spike data) for each branch input
    branch_inputs = []
    for i in range(N_BRANCHES):
        phase_i = delta_phase_list[i][:, np.newaxis, :]  # Shape: (n_samples, 1, time_dim)
        spk_i = spk_X_list[i][:, np.newaxis, :]          # Shape: (n_samples, 1, time_dim)
        branch_input = np.concatenate([phase_i, spk_i], axis=1)  # Shape: (n_samples, 2, time_dim)
        branch_inputs.append(branch_input)

    # Final input X shape: (n_samples, n_branches, 2, time_dim)
    X = np.stack(branch_inputs, axis=1)
    print(f"Finished data loading and preprocessing. X shape: {X.shape}")

    # Convert to float32 as expected by PyTorch
    X = X.astype(np.float32)
    Y_coords = Y_coords.astype(np.float32)

    return X, Y_coords, Y_original_labels