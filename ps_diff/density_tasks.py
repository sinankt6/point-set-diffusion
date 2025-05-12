import numpy as np
from typing import Dict, List

from ps_diff.metrics import (
    MMD,
    lengths_distribution_wasserstein_distance,
    points_distribution_wasserstein_distance,
)


def log_metrics(
    metrics_dict: Dict,
    logger,
    split: str,
    batch_size: int,
) -> None:
    """
    Logs the metrics in the metrics_dict to the logger.

    Args:
        metrics_dict (Dict): Dictionary with the stored metrics
        logger: logger to log the metrics
    """
    for metric in metrics_dict:
        logger.log(
            f"{split}/{metric}",
            metrics_dict[metric],
            batch_size=batch_size,
            on_epoch=True,
        )


def get_global_metrics(
    sample: List[np.ndarray],
    ground_truth: List[np.ndarray],
    space_bound: np.ndarray,
    n_max: int,
) -> Dict:
    metric_dict = {}

    # Compute metrics
    metric_dict["count_wasserstein"] = lengths_distribution_wasserstein_distance(
        ground_truth,
        sample,
        n_max,
    )
    metric_dict["point_wasserstein"] = points_distribution_wasserstein_distance(
        ground_truth,
        sample,
    )
    metric_dict["mmd_projection"] = MMD(
        X=ground_truth,
        Y=sample,
        distance="projection",
        space_bound=space_bound,
        dim=ground_truth[0].shape[-1],
    )[0]
    return metric_dict


def get_metrics_spp(
    sample: List[np.ndarray],
    ground_truth: List[np.ndarray],
    space_bound: np.ndarray,
    n_max: int,
    final_eval: bool = True,
) -> Dict:
    """
    Compute metrics for the SPP density estimation task.

    Assumes that the points are normalized to [-1, 1].

    Args:
        sampled (List[np.ndarray]): List of sampled sequences
        real (List[np.ndarray]): List of original sequences
        space_bound (np.ndarray): Space bound of the sequences
        n_max (int): Maximum number of points in the sequences
        final_eval (bool): Whether to compute final evaluation metrics

    Returns:
        metrics_dict (Dict): Dictionary containing the metrics

    """
    assert (space_bound[:, 0] == -1).all() and (
        space_bound[:, 1] == 1
    ).all(), "Space bound must be [-1, 1]"
    assert (np.concatenate(sample) <= 1).all() and (
        np.concatenate(sample) >= -1
    ).all(), "Sample must be normalized to [-1, 1]"
    assert (np.concatenate(ground_truth) <= 1).all() and (
        np.concatenate(ground_truth) >= -1
    ).all(), "ground_truth must be normalized to [-1, 1]"

    metric_dict = get_global_metrics(sample, ground_truth, space_bound, n_max)

    if final_eval:
        # MMD on wasserstein distance of points
        metric_dict["mmd_wasserstein"] = MMD(
            X=sample,
            Y=ground_truth,
            distance="wasserstein",
            dim=ground_truth[0].shape[-1],
            space_bound=space_bound,
        )[0]

    return metric_dict


def get_metrics_stpp(
    sample: List[np.ndarray],
    ground_truth: List[np.ndarray],
    space_bound: np.ndarray,
    n_max: int,
) -> Dict:
    """
    Compute metrics for the STPP forecasting task, where sample
    and ground_truth are in one-to-one correspondence.

    Assumes that the points are normalized to [-1, 1] and ordered by time.

    Args:
        sample (List[np.ndarray]): List of sampled sequences
        ground_truth (List[np.ndarray]): List of original sequences
        space_bound (np.ndarray): Space bound of the sequences
        n_max (int): Maximum number of points in the sequences

    Returns:
        metrics_dict (Dict): Dictionary containing the metrics
    """

    assert (space_bound[:, 0] == -1).all() and (
        space_bound[:, 1] == 1
    ).all(), "Space bound must be normalized to [-1, 1]"
    assert (np.concatenate(sample) <= 1).all() and (
        np.concatenate(sample) >= -1
    ).all(), "Sample must be normalized to [-1, 1]"
    assert (np.concatenate(ground_truth) <= 1).all() and (
        np.concatenate(ground_truth) >= -1
    ).all(), "ground_truth must be normalized to [-1, 1]"

    # Initialize dictionary
    metrics_dict = get_global_metrics(sample, ground_truth, space_bound, n_max)

    # Counting distances for STPPs
    # MMD counting distance for TPP
    metrics_dict["mmd_counting_tpp"] = MMD(
        X=ground_truth,
        Y=sample,
        distance="counting_tpp",
        dim=ground_truth[0].shape[-1],
        space_bound=space_bound,
    )[0]
    # MMD counting distance for SPP
    metrics_dict["mmd_counting_spp"] = MMD(
        X=ground_truth,
        Y=sample,
        distance="counting_spp",
        dim=ground_truth[0].shape[-1],
        space_bound=space_bound,
    )[0]
    # MMD counting distance for STPP
    metrics_dict["mmd_counting_stpp"] = MMD(
        X=ground_truth,
        Y=sample,
        distance="counting_stpp",
        dim=ground_truth[0].shape[-1],
        space_bound=space_bound,
    )[0]

    return metrics_dict
