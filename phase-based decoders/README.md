# Spatial Probe Decoding from LFP and Spike Data

A project to decode the spatial location of attentional focus from neural signals using a multi-branch deep learning model in PyTorch. The model fuses LFP phase information and spike data to predict 2D coordinates.

---

## Key Features

- **Multi-Modal Fusion**: Combines LFP delta-phase and raw spike data as input channels.
- **Multi-Branch Architecture**: Employs a dedicated `BranchModule` for each of the 16 V4 channels before fusing features for a final prediction.
- **End-to-End Workflow**: Includes scripts for data preprocessing, model training, evaluation, and visualization.
- **Reproducibility**: The experiment is run for multiple iterations with different random seeds, and performance is averaged for robust evaluation.
- **Comprehensive Evaluation**: Automatically generates performance metrics (RMSE, MAE, R²) and visualizations, including scatter plots, heatmaps, and CDFs of the prediction error.

---

## Model Architecture

The core of this project is the `MultiBranchEEGNet`, a neural network designed to handle multi-channel neurophysiological data. The CNN structure is inspired by the compact and efficient design of **EEGNet**, particularly its use of depthwise and separable convolutions to capture temporal and spatial features effectively.

1.  **Branch Module**: Each of the 16 input channels (16ch V4 LFP phase - single ch FEF LFP phase, 16ch V4 spike) is first processed by an independent `BranchModule`. This module uses a temporal convolution to extract time-varying features, followed by a depthwise convolution to fuse the LFP phase and spike data streams. This two-step process is analogous to the depthwise separable convolutions in EEGNet, allowing the model to learn channel-specific features efficiently.

2.  **Fusion Layer**: The outputs from all 16 branches are stacked and fed into a standard `Conv2d` layer. This layer's kernel spans all 16 branches, acting as a spatial filter that learns patterns across the entire V4 array to create a unified feature representation.

3.  **Output Layer**: The fused features are passed through pooling and fully connected layers to regress the final 2D coordinates of the attentional focus.

### Reference

The model's convolutional design principles are adapted from:

-   V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, **"EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces,"** *Journal of Neural Engineering*, vol. 15, no. 5, p. 056013, 2018.
    -   [Link to paper](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c)
---

## Directory Structure

```
.
├── sample_data/            # Sample data directory
│   ├── lfp/                # LFP .mat files (lfp_1.mat, ..., lfp_17.mat)
│   └── spk/                # Spike .mat files (spk_1.mat, ..., spk_17.mat)
├── Results/                # Output directory for plots and metrics
├── config.py               # Main configuration file for paths, model, and training params
├── data_utils.py           # Data loading and preprocessing functions
├── model.py                # PyTorch model definitions (MultiBranchEEGNet)
├── plotting_utils.py       # Functions for generating all visualizations
├── main.py                 # The main script to run the experiment
└── requirements.txt        # List of required Python packages
```
---

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## How to Run

1.  **Configure the Experiment (Optional):**
    Open `config.py` to modify parameters such as `EXPERIMENT_ID`, `N_RUNS`, learning rate, batch size, or data segmentation windows. The default settings are configured to run with the provided `sample_data`.

2.  **Run the Main Script:**
    Execute the `main.py` script from the root directory of the project.
    ```bash
    python main.py
    ```

---

## Output

The script will generate a new folder inside the `Results/` directory named according to the `EXPERIMENT_ID` in `config.py`. This folder will contain:

-   **Plots**: PNG images for each run and for the global average performance.
-   **TensorBoard Logs**: A `tensorboard_logs/` subdirectory with training and validation loss curves.
-   **Excel Metrics**: A `.xlsx` file containing detailed performance metrics.
