import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr # Should be imported in main or passed if not used here directly

# Note: R2 score, MSE, MAE calculations are done in main.py.
# These functions primarily focus on visualization.

def _setup_plot_style(title, xlabel, ylabel, xlim, ylim, xticks, yticks, grid=True, invert_y=False):
    """Helper function to set up common plot elements."""
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
    if xticks is not None: plt.xticks(xticks)
    if yticks is not None: plt.yticks(yticks)
    if grid: plt.grid(True)
    if invert_y: plt.gca().invert_yaxis()

def plot_scatter_predictions(true_coords, pred_coords, identifier_str, result_dir,
                             axis_limits, axis_ticks, plot_type="Run"):
    """
    Plots a scatter of true vs. predicted locations.

    Args:
        true_coords (np.ndarray): Array of true coordinates (n_samples, 2).
        pred_coords (np.ndarray): Array of predicted coordinates (n_samples, 2).
        identifier_str (str): String to identify the plot (e.g., "Run 1", "Global Average").
        result_dir (str): Directory to save the plot.
        axis_limits (tuple): Limits for X and Y axes (min, max).
        axis_ticks (list): Ticks for X and Y axes.
        plot_type (str): "Run" or "Global" to slightly adjust title and filename.
    """
    plt.figure(figsize=(6, 6))
    # true_coords[:, 0] is Row Index (X-axis), true_coords[:, 1] is Col Index (Y-axis)
    plt.scatter(true_coords[:, 0], true_coords[:, 1], color='blue', label="True Location", alpha=0.5)
    plt.scatter(pred_coords[:, 0], pred_coords[:, 1], color='red', label="Predicted Location", alpha=0.5)
    for i in range(len(true_coords)):
        plt.plot([true_coords[i, 0], pred_coords[i, 0]],
                 [true_coords[i, 1], pred_coords[i, 1]], 'k--', alpha=0.3)

    title_prefix = "Probe Location: True vs Predicted" if plot_type == "Global" else "Scatter Plot"
    _setup_plot_style(f"{title_prefix} ({identifier_str})",
                      "X Coordinate (Row Index from mapping)",
                      "Y Coordinate (Col Index from mapping)",
                      axis_limits, axis_limits, axis_ticks, axis_ticks)
    plt.legend()
    filename = f"scatter_{identifier_str.lower().replace(' ', '_')}.png"
    if plot_type == "Global":
        filename = f"global_scatter_average_trials.png" # Specific name for this global plot
    save_path = os.path.join(result_dir, filename)
    plt.savefig(save_path)
    plt.close()


def plot_overall_probe_error_heatmap(true_coords_all, pred_coords_all, identifier_str, result_dir,
                                     axis_ticks, plot_type="Run"):
    """
    Plots a heatmap of mean Euclidean error for each probe location.

    Args:
        true_coords_all (np.ndarray): All true coordinates for the set (n_samples, 2).
        pred_coords_all (np.ndarray): All predicted coordinates for the set (n_samples, 2).
        identifier_str (str): String to identify the plot.
        result_dir (str): Directory to save the plot.
        axis_ticks (list): Ticks for X and Y axes (for heatmap labels).
        plot_type (str): "Run" or "Global".
    """
    unique_locs_coords = np.unique(true_coords_all, axis=0) # [Row Index, Col Index]
    probe_errors = {}
    for loc_coord in unique_locs_coords:
        mask = np.all(true_coords_all == loc_coord, axis=1)
        if np.sum(mask) > 0:
            errors = np.linalg.norm(pred_coords_all[mask] - true_coords_all[mask], axis=1)
            probe_errors[tuple(loc_coord)] = np.mean(errors)

    error_grid = np.full((len(axis_ticks), len(axis_ticks)), np.nan) # Assuming 3x3 grid
    for (r_coord, c_coord), err_val in probe_errors.items():
        # r_coord = Row Index, c_coord = Col Index
        if 0 <= int(r_coord) < len(axis_ticks) and 0 <= int(c_coord) < len(axis_ticks):
            error_grid[int(r_coord), int(c_coord)] = err_val
        else:
            print(f"Warning: Coordinate ({r_coord}, {c_coord}) out of bounds for error_grid of size ({len(axis_ticks)}, {len(axis_ticks)}).")


    plt.figure(figsize=(6, 5))
    # error_grid is indexed by [Row_Index, Col_Index]
    # sns.heatmap(error_grid.T, ...) means:
    #   - heatmap x-axis displays Row_Index
    #   - heatmap y-axis displays Col_Index
    #   - ax.invert_yaxis() makes Col_Index 0 at top.
    ax = sns.heatmap(error_grid.T, annot=True, cmap="coolwarm",
                     cbar_kws={'label': 'Mean Euclidean Error'},
                     vmin=0, vmax=np.nanmax(error_grid) if not np.all(np.isnan(error_grid)) else 1,
                     xticklabels=axis_ticks, yticklabels=axis_ticks)

    title_prefix = "Mean Euclidean Prediction Error per Probe" if plot_type == "Global" else "Overall Probe Error Heatmap"
    _setup_plot_style(f"{title_prefix} ({identifier_str})",
                      "X Coordinate (Row Index from mapping)",
                      "Y Coordinate (Col Index from mapping)",
                      None, None, axis_ticks, axis_ticks, grid=False, invert_y=True) # Invert Y for heatmap
    filename = f"overall_probe_error_heatmap_{identifier_str.lower().replace(' ', '_')}.png"
    if plot_type == "Global":
         filename = f"global_overall_probe_error_heatmap_average.png"
    save_path = os.path.join(result_dir, filename)
    plt.savefig(save_path)
    plt.close()
    return probe_errors # Return for metrics


def plot_cdf_errors(true_coords, pred_coords, identifier_str, result_dir, plot_type="Run"):
    """
    Plots the Cumulative Distribution Function (CDF) of prediction errors.
    """
    if len(pred_coords) == 0: return
    errors_arr = np.linalg.norm(pred_coords - true_coords, axis=1)
    sorted_errors = np.sort(errors_arr)
    cdf_values = np.arange(1, len(errors_arr) + 1) / len(errors_arr)

    plt.figure(figsize=(6, 4))
    plt.plot(sorted_errors, cdf_values, marker="o", linestyle="-", color="b")
    title_prefix = "CDF of Prediction Errors"
    _setup_plot_style(f"{title_prefix} ({identifier_str})",
                      "Prediction Error (Euclidean Distance)",
                      "Cumulative Probability",
                      None, None, None, None)
    filename = f"cdf_{identifier_str.lower().replace(' ', '_')}.png"
    if plot_type == "Global":
        filename = f"global_cdf_average_errors.png"
    save_path = os.path.join(result_dir, filename)
    plt.savefig(save_path)
    plt.close()


def plot_prediction_error_distribution_per_probe(true_coords_all, pred_coords_all, identifier_str, result_dir,
                                               axis_limits, axis_ticks, plot_type="Run"):
    """
    For each true probe location, plots a KDE or scatter of predictions.
    Color-codes points by error if not using KDE.
    For Global plot, also shows center of average predictions and distance.
    """
    unique_locs_coords = np.unique(true_coords_all, axis=0) # [Row Index, Col Index]

    for probe_coord in unique_locs_coords: # probe_coord is [Row_Idx, Col_Idx]
        mask = np.all(true_coords_all == probe_coord, axis=1)
        if np.sum(mask) < 1:
            continue

        current_predictions = pred_coords_all[mask] # Shape (n_points_for_probe, 2)
        current_true_repeated = true_coords_all[mask] # Shape (n_points_for_probe, 2)
        current_errors = np.linalg.norm(current_predictions - current_true_repeated, axis=1)

        plt.figure(figsize=(8, 6))
        # probe_coord[0] (Row Index) is X, probe_coord[1] (Col Index) is Y
        plt.scatter(probe_coord[0], probe_coord[1], color='blue', marker='x', s=100, label="True Probe Location")

        use_kde = np.sum(mask) > 1 and len(np.unique(current_predictions[:, 0])) > 1 and len(np.unique(current_predictions[:, 1])) > 1
        if use_kde:
            # current_predictions[:, 0] (Pred Row Index) is X for KDE
            # current_predictions[:, 1] (Pred Col Index) is Y for KDE
            sns.kdeplot(x=current_predictions[:, 0], y=current_predictions[:, 1], cmap="coolwarm",
                        fill=True, alpha=0.6, thresh=0.05, levels=5, warn_singular=False)
            scatter_label = "Predicted Locations (KDE)"
        else:
            plt.scatter(current_predictions[:, 0], current_predictions[:, 1], color='red',
                        label="Predicted Location(s)")
            scatter_label = "Predicted Location(s)"


        # Scatter plot of individual predictions, color-coded by error
        scatter_plot = plt.scatter(x=current_predictions[:, 0], y=current_predictions[:, 1],
                                   c=current_errors, cmap="coolwarm", s=np.maximum(current_errors * 50, 10), # Adjusted size
                                   edgecolor="black", vmin=0, label=scatter_label if not use_kde else None) # Avoid duplicate label
        cbar = plt.colorbar(scatter_plot)
        cbar.set_label("Prediction Error (Euclidean Distance)")

        title_detail = f"Probe ({probe_coord[0]}, {probe_coord[1]}) ({identifier_str})"
        title_prefix = "Global Avg. Prediction Error Dist." if plot_type == "Global" else "Prediction Error Dist."

        if plot_type == "Global" and np.sum(mask) > 0:
            center_of_avg_preds = np.mean(current_predictions, axis=0)
            plt.scatter(center_of_avg_preds[0], center_of_avg_preds[1],
                        color='black', marker='*', s=200, label='Center of Avg. Predictions')
            distance_to_center = np.linalg.norm(center_of_avg_preds - probe_coord)
            plt.plot([probe_coord[0], center_of_avg_preds[0]],
                     [probe_coord[1], center_of_avg_preds[1]],
                     linestyle="--", color="black", linewidth=2)
            plt.text(0.95, 0.05, f"Dist to Center: {distance_to_center:.2f}", # Adjusted position
                     color="black", fontsize=12, ha='right', va='bottom',
                     transform=plt.gca().transAxes,
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        _setup_plot_style(f"{title_prefix} for {title_detail}",
                          "X Coordinate (Row Index from mapping)",
                          "Y Coordinate (Col Index from mapping)",
                          axis_limits, axis_limits, axis_ticks, axis_ticks)
        plt.legend()


        probe_str = f"R{int(probe_coord[0])}_C{int(probe_coord[1])}"
        filename_prefix = "global_avg_heatmap_probe_" if plot_type == "Global" else f"heatmap_probe_"
        filename = f"{filename_prefix}{probe_str}_{identifier_str.lower().replace(' ', '_')}.png"

        save_path = os.path.join(result_dir, filename)
        plt.savefig(save_path)
        plt.close()


def plot_average_predicted_location_per_probe(true_coords_all, pred_coords_all, identifier_str, result_dir,
                                              axis_limits, axis_ticks, plot_type="Run"):
    """
    Plots true probe locations and the average predicted location for each.
    """
    unique_true_locs = np.unique(true_coords_all, axis=0) # These are [Row Index, Col Index]
    if len(unique_true_locs) == 0: return

    avg_pred_locations_for_true = []
    for true_loc in unique_true_locs:
        mask = np.all(true_coords_all == true_loc, axis=1)
        if np.sum(mask) > 0:
            avg_pred_loc = np.mean(pred_coords_all[mask], axis=0)
            avg_pred_locations_for_true.append(avg_pred_loc)
    avg_pred_locations_for_true = np.array(avg_pred_locations_for_true)

    if len(avg_pred_locations_for_true) == 0: return

    plt.figure(figsize=(6, 6))
    # unique_true_locs[:, 0] (Row Index) is X, unique_true_locs[:, 1] (Col Index) is Y
    plt.scatter(unique_true_locs[:, 0], unique_true_locs[:, 1], color='blue',
                label="True Probe Location", alpha=0.5, s=100)

    pred_label = "Global Avg. Predicted Location per Probe" if plot_type == "Global" else "Average Predicted Location"
    pred_color = 'magenta' if plot_type == "Global" else 'green'
    plt.scatter(avg_pred_locations_for_true[:, 0], avg_pred_locations_for_true[:, 1], color=pred_color,
                label=pred_label, alpha=0.7, marker='s', s=100)

    for i in range(len(unique_true_locs)):
        plt.plot([unique_true_locs[i, 0], avg_pred_locations_for_true[i, 0]],
                 [unique_true_locs[i, 1], avg_pred_locations_for_true[i, 1]], 'k--', alpha=0.3)

    title_prefix = "Global Average Predicted Location per Probe" if plot_type == "Global" else "Average Predicted Location per Probe"
    _setup_plot_style(f"{title_prefix} ({identifier_str})",
                      "X Coordinate (Row Index from mapping)",
                      "Y Coordinate (Col Index from mapping)",
                      axis_limits, axis_limits, axis_ticks, axis_ticks)
    plt.legend()

    filename = f"avg_probe_scatter_{identifier_str.lower().replace(' ', '_')}.png"
    if plot_type == "Global":
        filename = f"global_avg_predicted_location_per_probe.png"
    save_path = os.path.join(result_dir, filename)
    plt.savefig(save_path)
    plt.close()