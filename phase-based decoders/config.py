import torch
import os

# -----------------------------
# Basic Configuration
# -----------------------------
# This ID is used to create a unique folder for the results and logs of this experiment.
EXPERIMENT_ID = "sample_data_experiment"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_RUNS = 5  # Number of times to repeat the training and evaluation
RANDOM_STATE_BASE = 0 # Base for random state in train_test_split for reproducibility across runs

# -----------------------------
# Data Paths (Relative to the main script)
# -----------------------------
# V4 LFP and Spike data (Channels 1-16)
LFP_DATA_PATH_TEMPLATE = os.path.join('sample_data', 'lfp', 'lfp_{channel_id}.mat')
SPIKE_DATA_PATH_TEMPLATE = os.path.join('sample_data', 'spk', 'spk_{channel_id}.mat')

# Define which channel IDs correspond to FEF and the source for condition labels
# Based on the new structure, lfp_17.mat and spk_17.mat seem to be FEF data.
# We assume conditions are stored in one of the LFP files (e.g., lfp_1.mat).
FEF_LFP_CHANNEL_ID = 17
CONDITION_SOURCE_CHANNEL_ID = 1


# -----------------------------
# Preprocessing Parameters
# -----------------------------
LFP_LOWCUT = 4      # Hz
LFP_HIGHCUT = 30    # Hz
LFP_FS = 1000       # Sampling frequency in Hz
LFP_FILTER_ORDER = 5

# LFP segmentation: 1001ms to 1300ms (inclusive) from the 1500ms data
LFP_SEGMENT_START_MS = 1000 # 0-indexed, so 1000 is the 1001st ms
LFP_SEGMENT_END_MS = 1300   # End index for slicing (exclusive), so captures up to 1299th index

# Spike segmentation: first 300ms (index 0 to 299)
SPIKE_SEGMENT_START_MS = 0
SPIKE_SEGMENT_END_MS = 300

N_LFP_CHANNELS = 16 # V4 LFP channels
N_SPIKE_CHANNELS = 16 # V4 Spike channels

# -----------------------------
# Model & Training Parameters
# -----------------------------
OUTPUT_SIZE = 2  # For X and Y coordinates
N_BRANCHES = 16
MAX_EPOCHS = 200
LEARNING_RATE = 0.00002
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 16
VALIDATION_SPLIT_RATIO = 0.1 # 10% of training data for validation
TEST_SPLIT_RATIO = 0.1      # 10% of total data for testing

# -----------------------------
# Output and Logging
# -----------------------------
RESULTS_BASE_DIR = os.path.join("Results") # Base for saving results
# Specific result directory will be RESULTS_BASE_DIR / EXPERIMENT_ID
# Tensorboard logs will be RESULTS_BASE_DIR / EXPERIMENT_ID / "tensorboard_logs"

# -----------------------------
# Plotting Parameters
# -----------------------------
# Mapping of condition labels to actual probe spatial [Row, Column] coordinates
# Row index corresponds to X-coordinate in scatter plots, Col index to Y-coordinate
PROBE_MAPPING = {
    56: [0, 2], 55: [1, 2], 54: [2, 2],
    57: [0, 1], 51: [1, 1], 53: [2, 1],
    58: [0, 0], 59: [1, 0], 52: [2, 0]
}
# Axis limits and ticks for consistent plotting [X_min, X_max] or [Y_min, Y_max]
# Corresponds to Row/Col indices from PROBE_MAPPING
AXIS_LIMITS = (-0.25, 2.25)
AXIS_TICKS = [0, 1, 2]

# -----------------------------
# Excel Output
# -----------------------------
EXCEL_FILENAME_TEMPLATE = "{experiment_id}_performance_metrics_100ms_LFP_300ms_Spike.xlsx"