import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from skorch import NeuralNetRegressor
from skorch.dataset import ValidSplit

# Import from local modules
import config
from data_utils import load_and_preprocess_data
from model import MultiBranchEEGNet, TensorBoardCallback
import plotting_utils as pu


def main():
    """
    Main function to run the experiment.
    """
    start_run_time = time.time()

    # --- Setup Directories ---
    # Specific result directory for this experiment, using EXPERIMENT_ID
    current_experiment_result_dir = os.path.join(config.RESULTS_BASE_DIR, config.EXPERIMENT_ID)
    os.makedirs(current_experiment_result_dir, exist_ok=True)

    # Base directory for TensorBoard logs for this experiment
    experiment_log_dir_base = os.path.join(current_experiment_result_dir, "tensorboard_logs")
    os.makedirs(experiment_log_dir_base, exist_ok=True)

    # --- Load and Preprocess Data ---
    # Paths are now constructed using templates from config
    X, Y_coords, Y_original_labels = load_and_preprocess_data(
        probe_mapping=config.PROBE_MAPPING,
        lfp_path_template=config.LFP_DATA_PATH_TEMPLATE,
        spike_path_template=config.SPIKE_DATA_PATH_TEMPLATE
    )

    N_total_samples = X.shape[0]
    print(f"Total samples loaded: {N_total_samples}")
    print(f"Input data X shape: {X.shape}, Target Y_coords shape: {Y_coords.shape}")
    print(f"Original labels Y shape (for stratification): {Y_original_labels.shape}")

    # --- Training and Evaluation Loop (Multiple Runs) ---
    all_runs_metrics = []
    all_runs_probe_errors_list = []  # For per-run, per-probe errors

    # For global average predictions across all test samples over runs
    global_predictions_aggregator = {}  # Key: original sample index, Value: list of predictions
    global_groundtruth_map = {}  # Key: original sample index, Value: ground truth Y_coord

    for run_idx in range(config.N_RUNS):
        run_id_str = f"Run {run_idx + 1}"
        print(f"\n--- Starting {run_id_str}/{config.N_RUNS} ---")

        # Split data for the current run
        # Stratify by original labels (Y_original_labels) to ensure class balance
        # Use a different random_state for each run to get different splits
        train_indices, test_indices = train_test_split(
            np.arange(N_total_samples),
            test_size=config.TEST_SPLIT_RATIO,
            stratify=Y_original_labels,  # Use original (non-coordinate) labels for stratification
            random_state=config.RANDOM_STATE_BASE + run_idx,
            shuffle=True
        )

        trainX_run = X[train_indices]
        trainY_run = Y_coords[train_indices]
        testX_run = X[test_indices]
        testY_run = Y_coords[test_indices]
        testY_original_labels_run = Y_original_labels[test_indices]  # For checking test set distribution

        unique_test_labels, counts_test_labels = np.unique(testY_original_labels_run, return_counts=True)
        print(f"{run_id_str} Test set original label distribution: {dict(zip(unique_test_labels, counts_test_labels))}")
        print(f"{run_id_str} Train set size: {len(trainX_run)}, Test set size: {len(testX_run)}")

        # TensorBoard log directory for the current run
        current_run_tb_log_dir = os.path.join(experiment_log_dir_base, f"run_{run_idx + 1}")
        os.makedirs(current_run_tb_log_dir, exist_ok=True)

        # Initialize and train the model for the current run
        # TIME_DIM is now implicitly handled by model.py based on config
        net = NeuralNetRegressor(
            module=MultiBranchEEGNet,
            module__outputSize=config.OUTPUT_SIZE,
            module__n_branches=config.N_BRANCHES,
            # module__time_dim can be passed if it's not fixed or derived in model.py
            max_epochs=config.MAX_EPOCHS,
            lr=config.LEARNING_RATE,
            optimizer=optim.Adam,
            optimizer__weight_decay=config.WEIGHT_DECAY,
            criterion=nn.MSELoss,
            batch_size=config.BATCH_SIZE,
            iterator_train__shuffle=True,
            device=config.DEVICE,
            train_split=ValidSplit(cv=config.VALIDATION_SPLIT_RATIO, stratified=False),
            # Skorch handles validation split
            callbacks=[TensorBoardCallback(log_dir=current_run_tb_log_dir)]
        )

        print(f"Training model for {run_id_str}...")
        net.fit(trainX_run, trainY_run)

        # Evaluate on the test set for the current run
        y_pred_run = net.predict(testX_run)

        # Aggregate predictions for global analysis
        for i, original_sample_idx in enumerate(test_indices):
            if original_sample_idx not in global_predictions_aggregator:
                global_predictions_aggregator[original_sample_idx] = []
                global_groundtruth_map[original_sample_idx] = Y_coords[original_sample_idx]
            global_predictions_aggregator[original_sample_idx].append(y_pred_run[i])

        # Calculate metrics for the current run
        mse_run = mean_squared_error(testY_run, y_pred_run)
        rmse_run = np.sqrt(mse_run)
        mae_run = mean_absolute_error(testY_run, y_pred_run)
        r2_overall_run = r2_score(testY_run, y_pred_run)

        r2_x_run, pearson_x_run = np.nan, np.nan
        if len(np.unique(testY_run[:, 0])) > 1 and len(np.unique(y_pred_run[:, 0])) > 1:
            r2_x_run = r2_score(testY_run[:, 0], y_pred_run[:, 0])
            pearson_x_run, _ = pearsonr(testY_run[:, 0], y_pred_run[:, 0])

        r2_y_run, pearson_y_run = np.nan, np.nan
        if len(np.unique(testY_run[:, 1])) > 1 and len(np.unique(y_pred_run[:, 1])) > 1:
            r2_y_run = r2_score(testY_run[:, 1], y_pred_run[:, 1])
            pearson_y_run, _ = pearsonr(testY_run[:, 1], y_pred_run[:, 1])

        pearson_avg_run = np.nanmean([pearson_x_run, pearson_y_run])

        current_run_metrics = {
            "Run": run_idx + 1, "MSE": mse_run, "RMSE": rmse_run, "MAE": mae_run,
            "R^2 (overall)": r2_overall_run, "R^2_x": r2_x_run, "R^2_y": r2_y_run,
            "Pearson_X": pearson_x_run, "Pearson_Y": pearson_y_run, "Pearson_avg": pearson_avg_run
        }
        all_runs_metrics.append(current_run_metrics)
        print(f"{run_id_str} - MSE: {mse_run:.4f}, RMSE: {rmse_run:.4f}, MAE: {mae_run:.4f}, R²: {r2_overall_run:.4f}")

        # --- Per-Run Visualizations ---
        print(f"Generating plots for {run_id_str}...")
        pu.plot_scatter_predictions(testY_run, y_pred_run, run_id_str, current_experiment_result_dir,
                                    config.AXIS_LIMITS, config.AXIS_TICKS, plot_type="Run")

        run_probe_errors = pu.plot_overall_probe_error_heatmap(testY_run, y_pred_run, run_id_str,
                                                               current_experiment_result_dir,
                                                               config.AXIS_TICKS, plot_type="Run")
        for loc_coord_tuple, mean_err in run_probe_errors.items():
            all_runs_probe_errors_list.append({
                "Run": run_idx + 1,
                "Probe_Location": str(loc_coord_tuple),  # Ensure it's a string for DataFrame
                "Mean_Euclidean_Error": mean_err
            })

        pu.plot_cdf_errors(testY_run, y_pred_run, run_id_str, current_experiment_result_dir, plot_type="Run")

        pu.plot_prediction_error_distribution_per_probe(testY_run, y_pred_run, run_id_str, current_experiment_result_dir,
                                                        config.AXIS_LIMITS, config.AXIS_TICKS, plot_type="Run")

        pu.plot_average_predicted_location_per_probe(testY_run, y_pred_run, run_id_str, current_experiment_result_dir,
                                                     config.AXIS_LIMITS, config.AXIS_TICKS, plot_type="Run")
        print(f"Finished plots for {run_id_str}.")

    # --- Global Average Performance (across all unique test samples over runs) ---
    print("\n--- Calculating Global Average Performance ---")
    global_avg_preds_list = []
    global_truth_list = []

    # Ensure original indices are sorted for consistent ordering if needed later
    sorted_original_indices_in_test = sorted(list(global_predictions_aggregator.keys()))

    for original_idx in sorted_original_indices_in_test:
        pred_list_for_idx = np.array(global_predictions_aggregator[original_idx])
        avg_pred_for_idx = np.mean(pred_list_for_idx, axis=0)
        global_avg_preds_list.append(avg_pred_for_idx)
        global_truth_list.append(global_groundtruth_map[original_idx])

    global_avg_preds_arr = np.array(global_avg_preds_list)
    global_truth_arr = np.array(global_truth_list)  # True coordinates are [Row Index, Col Index]

    mse_avg_global, rmse_avg_global, mae_avg_global = np.nan, np.nan, np.nan
    r2_overall_avg_global, r2_x_avg_global, r2_y_avg_global = np.nan, np.nan, np.nan
    pearson_x_avg_global, pearson_y_avg_global, pearson_avg_overall_global = np.nan, np.nan, np.nan
    avg_probe_errors_summary_global = {}

    if len(global_truth_arr) > 0:
        mse_avg_global = mean_squared_error(global_truth_arr, global_avg_preds_arr)
        rmse_avg_global = np.sqrt(mse_avg_global)
        mae_avg_global = mean_absolute_error(global_truth_arr, global_avg_preds_arr)
        r2_overall_avg_global = r2_score(global_truth_arr, global_avg_preds_arr)

        if len(np.unique(global_truth_arr[:, 0])) > 1 and len(np.unique(global_avg_preds_arr[:, 0])) > 1:
            r2_x_avg_global = r2_score(global_truth_arr[:, 0], global_avg_preds_arr[:, 0])
            pearson_x_avg_global, _ = pearsonr(global_truth_arr[:, 0], global_avg_preds_arr[:, 0])

        if len(np.unique(global_truth_arr[:, 1])) > 1 and len(np.unique(global_avg_preds_arr[:, 1])) > 1:
            r2_y_avg_global = r2_score(global_truth_arr[:, 1], global_avg_preds_arr[:, 1])
            pearson_y_avg_global, _ = pearsonr(global_truth_arr[:, 1], global_avg_preds_arr[:, 1])

        pearson_avg_overall_global = np.nanmean([pearson_x_avg_global, pearson_y_avg_global])

        print("\n=== Global Average Performance Metrics (on averaged predictions) ===")
        print(f"MSE: {mse_avg_global:.4f}, RMSE: {rmse_avg_global:.4f}, MAE: {mae_avg_global:.4f}")
        print(f"R² (overall): {r2_overall_avg_global:.4f}, R²_x: {r2_x_avg_global:.4f}, R²_y: {r2_y_avg_global:.4f}")
        print(
            f"Pearson X: {pearson_x_avg_global:.4f}, Pearson Y: {pearson_y_avg_global:.4f}, Avg Pearson: {pearson_avg_overall_global:.4f}")

        # Probe-specific errors based on global average predictions
        avg_probe_errors_summary_global = pu.plot_overall_probe_error_heatmap(global_truth_arr, global_avg_preds_arr,
                                                                              "Global Average",
                                                                              current_experiment_result_dir,
                                                                              config.AXIS_TICKS, plot_type="Global")
        print("\nProbe Mean Euclidean Error (Global Average Predictions):")
        for loc, err in avg_probe_errors_summary_global.items():
            print(f"Probe {loc}: {err:.4f}")

        # --- Global Average Visualizations ---
        print("Generating global average plots...")
        pu.plot_scatter_predictions(global_truth_arr, global_avg_preds_arr, "Global Average",
                                    current_experiment_result_dir,
                                    config.AXIS_LIMITS, config.AXIS_TICKS, plot_type="Global")
        pu.plot_cdf_errors(global_truth_arr, global_avg_preds_arr, "Global Average", current_experiment_result_dir,
                           plot_type="Global")
        pu.plot_average_predicted_location_per_probe(global_truth_arr, global_avg_preds_arr, "Global Average",
                                                     current_experiment_result_dir,
                                                     config.AXIS_LIMITS, config.AXIS_TICKS, plot_type="Global")
        pu.plot_prediction_error_distribution_per_probe(global_truth_arr, global_avg_preds_arr, "Global Average",
                                                        current_experiment_result_dir,
                                                        config.AXIS_LIMITS, config.AXIS_TICKS, plot_type="Global")
        print("Finished global average plots.")
    else:
        print("No global average predictions to evaluate or plot.")

    # --- Save Metrics to Excel ---
    print("\nSaving performance metrics to Excel...")
    metrics_df = pd.DataFrame(all_runs_metrics)
    if not metrics_df.empty:
        avg_metrics_row = metrics_df.mean(numeric_only=True).to_frame().T
        avg_metrics_row[
            "Run"] = "Average_of_Runs"  # Clarify this is average of run metrics, not global avg prediction metrics
        # metrics_df = pd.concat([metrics_df, avg_metrics_row], ignore_index=True) # Optional: add this average to the run metrics sheet
        # Keep df_overall_avg_metrics separate as it's calculated differently

    df_overall_avg_metrics = pd.DataFrame({
        "Metric": ["MSE", "RMSE", "MAE", "R^2 (overall)", "R^2_x", "R^2_y", "Pearson_X", "Pearson_Y", "Pearson_avg"],
        "Value": [mse_avg_global, rmse_avg_global, mae_avg_global, r2_overall_avg_global, r2_x_avg_global,
                  r2_y_avg_global,
                  pearson_x_avg_global, pearson_y_avg_global, pearson_avg_overall_global]
    })

    # Per-run probe errors
    df_probe_errors_runs = pd.DataFrame(all_runs_probe_errors_list)

    # Global average probe errors
    probe_list_global_avg = []
    mean_errors_list_global_avg = []
    for loc, err in avg_probe_errors_summary_global.items():
        probe_list_global_avg.append(str(loc))
        mean_errors_list_global_avg.append(err)
    df_probe_errors_global_avg = pd.DataFrame({
        "Probe_Location": probe_list_global_avg,
        "Mean_Euclidean_Error_GlobalAvgPreds": mean_errors_list_global_avg
    })

    excel_filename = config.EXCEL_FILENAME_TEMPLATE.format(experiment_id=config.EXPERIMENT_ID)
    excel_path = os.path.join(current_experiment_result_dir, excel_filename)
    with pd.ExcelWriter(excel_path) as writer:
        metrics_df.to_excel(writer, sheet_name="Run_Metrics", index=False)
        if not avg_metrics_row.empty:  # Save the average of run metrics if calculated
            avg_metrics_row.to_excel(writer, sheet_name="Average_of_Run_Metrics", index=False)
        df_overall_avg_metrics.to_excel(writer, sheet_name="Global_Avg_Prediction_Metrics", index=False)
        if not df_probe_errors_runs.empty:
            df_probe_errors_runs.to_excel(writer, sheet_name="Probe_Errors_Per_Run", index=False)
        if not df_probe_errors_global_avg.empty:
            df_probe_errors_global_avg.to_excel(writer, sheet_name="Probe_Errors_Global_Avg_Preds", index=False)
    print(f"Performance metrics saved to: {excel_path}")

    end_run_time = time.time()
    elapsed_time = end_run_time - start_run_time
    print(f"\nTotal execution time: {elapsed_time / 60:.2f} minutes ({elapsed_time:.2f} seconds)")
    print("Processing complete.")


if __name__ == '__main__':
    # Set a fixed seed for reproducibility of Pytorch operations if desired,
    # though sklearn's train_test_split random_state per run already helps.
    # torch.manual_seed(42)
    # np.random.seed(42)
    # if torch.cuda.is_available():
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    main()