from ot import sliced_wasserstein_distance
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os

from ps_diff.data import Sequence, Batch, LIMITS_MAPS, MAPS

from statistics import mean, median
from typing import Dict, List
from ps_diff.data import Batch, Sequence
from ps_diff.metrics import (
    counting_distance,
    mae_length_forecast_distance,
    projection_distance,
    match_shapes,
)

OPERATORS = {"mean": mean, "median": median, "min": min, "max": max}


def conditional_task_boxes(
    sequences: List[np.ndarray],
    space_bound: np.ndarray,
    n_boxes: int,
    n_original_sequences: int,
    seed: int = 80672983,
    dim: int = 2,
    min_box=0.5,
    max_diameter=1.5,
) -> Dict:
    """
    Generate spatial boxes for sequences to forecast.
    Assume that the sequence dimensions are all normalized to [-1, 1].

    Returns:
        sampled_dict (Dict): Dictionary containing the following:
            real_batch (Batch): batch with real data used for the forecasting
            sampled_batch (Batch): batch sampled during forecasting
            x_min (torch.Tensor): x_min of the forecasting area
            shift (torch.Tensor): shift on the forecasting
        global_x_min (torch.Tensor): Randomly sampled x_min
        global_shift (torch.Tensor): Randomly sampled shifts
    """
    for seq in sequences:
        assert seq.shape[1] == dim, "Sequences must be right dimension"

    assert n_original_sequences <= len(sequences), "Not enough sequences to sample from"

    # Make it reproducible
    np.random.seed(seed)

    # Subsample original sequences if n_original_sequences < total number of sequences
    orig_seq_ind = np.random.choice(len(sequences), n_original_sequences, replace=False)

    found = False

    while found is False:

        # Get boxes and shift # Width is in [0.5, 1.5] # old * (1.5 - 0.5) + 0.5
        width = np.random.uniform(
            low=min_box, high=max_diameter, size=(n_boxes, dim)
        ).astype(np.float32)

        start_value = np.random.rand(n_boxes, dim).astype(np.float32) * (2 - width) - 1

        x_max = start_value + width
        x_min = start_value

        conditional_task = dict()

        n_forecastes = torch.zeros(n_boxes, dtype=torch.int32)

        for ind in orig_seq_ind:
            # Create the batch with n_forecasted_sequences of the same real sequence
            batch = Batch.from_sequence_list(
                [
                    Sequence(sequences[ind], space_bound=space_bound)
                    for _ in range(n_boxes)
                ]
            )
            # Get forecasting information
            condition, forecast, _, _ = batch.split_space(
                torch.tensor(x_min), torch.tensor(x_max)
            )
            n_forecastes += forecast.mask[:, :, 0].sum(1)
            conditional_task[ind] = {
                "box": [(x_min[i], x_max[i]) for i in range(n_boxes)],
                "condition": condition.to_point_list(),
                "forecast": forecast.to_point_list(),
                "full_data": batch.to_point_list(),
            }
        found = (n_forecastes > 0).all()

    return conditional_task


def conditional_task_time(
    sequences: List[np.ndarray],
    space_bound: np.ndarray,
    n_forecasts: int,
    n_original_sequences: int,
    seed: int = 80672983,
    dim: int = 2,
    min_forecast=0.5,
) -> Dict:
    """
    Generate temporal forecasting task for sequences.
    Assume that the sequence dimensions are all normalized to [-1, 1].

    Returns:

    """
    for seq in sequences:
        assert seq.shape[1] == dim, "Sequences must be right dimension"

    assert n_original_sequences <= len(sequences), "Not enough sequences to sample from"

    # Make it reproducible
    np.random.seed(seed)

    # Subsample original sequences if n_original_sequences < total number of sequences
    orig_seq_ind = np.random.choice(len(sequences), n_original_sequences, replace=False)

    conditional_task = dict()

    for ind in orig_seq_ind:
        x_min = []
        x_max = []
        seq = sequences[ind]
        for i in range(n_forecasts):
            found = False
            # Check if there are at least 2 points in the history
            while not found:
                t_f = np.random.uniform(-1, 1 - min_forecast)
                found = (seq[:, 0] <= t_f).sum() >= 2
            # Define forecasting space
            x_min.append(np.array([[t_f, -1.0, -1.0]]).astype(np.float32))
            x_max.append(np.array([[1.0, 1.0, 1.0]]).astype(np.float32))

        x_min = np.concatenate(x_min, axis=0)
        x_max = np.concatenate(x_max, axis=0)

        # Create the batch with n_forecasted_sequences of the same real sequence
        batch = Batch.from_sequence_list(
            [Sequence(seq, space_bound=space_bound) for _ in range(n_forecasts)]
        )
        # Get forecasting information
        condition, forecast, _, _ = batch.split_space(
            torch.tensor(x_min), torch.tensor(x_max)
        )
        conditional_task[ind] = {
            "box": [(x_min[i], x_max[i]) for i in range(n_forecasts)],
            "condition": condition.to_point_list(),
            "forecast": forecast.to_point_list(),
            "full_data": batch.to_point_list(),
        }
    return conditional_task


def load_conditional_task(generation_task, space_bound, device="cuda"):
    forecasts = []
    conditions = []
    fulls = []

    x_mins, x_maxs = [], []

    for key in generation_task.keys():
        x_min, x_max = zip(*generation_task[key]["box"])
        x_mins = x_mins + list(x_min)
        x_maxs = x_maxs + list(x_max)

        condition = generation_task[key]["condition"]
        forecast = generation_task[key]["forecast"]
        full = generation_task[key]["full_data"]
        forecasts = forecasts + forecast
        conditions = conditions + condition
        fulls = fulls + full

    condition = Batch.from_sequence_list(
        [Sequence(seq, space_bound) for seq in conditions]
    ).to(device)
    forecasts = Batch.from_sequence_list(
        [Sequence(seq, space_bound) for seq in forecasts]
    ).to(device)
    fulls = Batch.from_sequence_list([Sequence(seq, space_bound) for seq in fulls]).to(
        device
    )
    x_mins = torch.tensor(np.array(x_mins), device=device)
    x_maxs = torch.tensor(np.array(x_maxs), device=device)

    return forecasts, condition, fulls, x_mins, x_maxs


def sample_points_for_id(
    i,
    real_list,
    condition_list,
    space_bound,
    x_mins_list,
    x_maxs_list,
    task,
    device="cuda",
):
    real = Batch.from_sequence_list(
        [Sequence(real_list[i], space_bound) for _ in range(1000)]
    ).to(device)
    condition = Batch.from_sequence_list(
        [Sequence(condition_list[i], space_bound) for _ in range(1000)]
    ).to(device)

    x_mins = [x_mins_list[i] for _ in range(1000)]
    x_maxs = [x_maxs_list[i] for _ in range(1000)]

    _, sampled_forecast = task.model.conditional_generation(
        condition,
        torch.tensor(x_mins, device=device).float(),
        torch.tensor(x_maxs, device=device).float(),
    )

    return (
        real.to_point_list(),
        sampled_forecast.to_point_list(),
        condition.to_point_list(),
        x_mins,
        x_maxs,
    )


################
### SPP TASK ###
################
def get_metrics_spp(
    conditional_sampled: List[np.ndarray],
    ground_truth: List[np.ndarray],
) -> Dict:
    """
    Compute metrics for the SPP conditional generation task, where conditional_sampled
    and ground_truth are in one-to-one correspondence.

    Assumes that the points are normalized to [-1, 1].

    Args:
        conditional_sampled (List[np.ndarray]): List of sampled sequences
        ground_truth (List[np.ndarray]): List of original sequences

    Returns:
        metrics_dict (Dict): Dictionary containing the metrics
    """
    assert (np.concatenate(conditional_sampled) <= 1).all() and (
        np.concatenate(conditional_sampled) >= -1
    ).all(), "Sample must be normalized to [-1, 1]"
    assert (np.concatenate(ground_truth) <= 1).all() and (
        np.concatenate(ground_truth) >= -1
    ).all(), "ground_truth must be normalized to [-1, 1]"

    # Initialize dictionary
    metrics_dict = {}

    # Metrics on length of sequences
    # MAE distance
    metrics_dict["mae"] = mae_length_forecast_distance(
        X=conditional_sampled,
        Y=ground_truth,
    )

    # Metrics on positions of points in sequence
    # Wasserstein distance
    wass_point = []
    for i in range(len(ground_truth)):
        # Only compute if there are points to compute
        if (conditional_sampled[i].shape[0] > 0) and (ground_truth[i].shape[0] > 0):
            wass_point.append(
                sliced_wasserstein_distance(conditional_sampled[i], ground_truth[i])
            )

    metrics_dict["wass_point"] = wass_point

    X, Y, X_mask, Y_mask = match_shapes(
        X=ground_truth,
        Y=conditional_sampled,
        dim=ground_truth[0].shape[-1],
    )

    proj = []
    for i in range(len(ground_truth)):
        # Only compute if there are points to compute
        if (X_mask[i].sum() > 0) or (Y_mask[i].sum() > 0):
            proj.append(
                projection_distance(
                    x=X[i],
                    Y=np.expand_dims(Y[i], axis=0),
                    x_mask=X_mask[i],
                    Y_mask=np.expand_dims(Y_mask[i], axis=0),
                    max_value=np.ones(X.shape[-1]).astype(np.float32),
                    dim=X.shape[-1],
                ).item()
            )
    metrics_dict["proj"] = proj
    return metrics_dict


def log_metrics_spp(
    metrics_dict: Dict,
    logger,
    split: str,
) -> None:
    """
    Computes the final metrics to log and logs them.

    Args:
        metrics_dict (Dict): Dictionary with the stored metrics
        logger: logger to log the metrics
    """
    # Log MAE
    logger.log({f"{split}/mae": metrics_dict["mae"].item()})

    for operator in OPERATORS:
        # Log wass_point as min
        logger.log(
            {
                f"{split}/{operator}_wass_point": OPERATORS[operator](
                    metrics_dict["wass_point"]
                )
            }
        )
        # Log proj distance as min
        logger.log(
            {f"{split}/{operator}_proj": OPERATORS[operator](metrics_dict["proj"])}
        )
