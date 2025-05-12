from typing import List

import numpy as np

from einops import reduce, repeat
from scipy.stats import wasserstein_distance
from ot import sliced_wasserstein_distance

from torchtyping import patch_typeguard
from typeguard import typechecked


patch_typeguard()


#########################################
### Utilities for Metric calculations ###
#########################################


@typechecked
def match_shapes(
    X: List[np.ndarray],
    Y: List[np.ndarray],
    dim: int,
):
    """
    Match shapes between two lists of np.ndarray. Returns two np.ndarray with the same
    length in the second dim.

    Assumes that the sequences are padded with 1.0, that is the maximum value for the
    given dimension.

    Args:
        X (List): List of sequences
        Y (List): List of sequences
        dim (int): Dimensions of the space

    Returns:
        new_X (np.array): Numpy array with adjusted length on second dim.
        new_Y (np.array): Numpy array with adjusted length on second dim.
        X_mask (np.array): Mask for padding in X
        Y_mask (np.array): Mask for padding in Y
    """

    # max length of sequences
    len_x = [x.shape[0] for x in X]
    len_y = [y.shape[0] for y in Y]

    max_x = max(len_x)
    max_y = max(len_y)
    max_size = max(max_x, max_y)

    # Get mask for padding
    X_mask = repeat(
        np.arange(0, max_size)[None, :] < np.array(len_x)[:, None],
        "b s -> b s dim",
        dim=dim,
    )
    Y_mask = repeat(
        np.arange(0, max_size)[None, :] < np.array(len_y)[:, None],
        "b s -> b s dim",
        dim=dim,
    )

    # Get padded sequences
    new_X = np.ones((len(X), max_size, dim))
    new_Y = np.ones((len(Y), max_size, dim))
    for i, x in enumerate(X):
        new_X[i, : len(x), :] = x
    for i, y in enumerate(Y):
        new_Y[i, : len(y), :] = y

    return new_X, new_Y, X_mask, Y_mask


@typechecked
def gaussian_kernel(x: np.ndarray, sigma2: float = 1):
    return np.exp(-x / (2 * sigma2))


###############################
###          Metrics        ###
###############################


@typechecked
def counting_distance(
    x: np.ndarray,
    Y: np.ndarray,
    x_mask: np.ndarray,
    Y_mask: np.ndarray,
    max_value: np.ndarray,
    mode: str,
):
    """
    Computes the distance between the counting process of x and y along the
    dimensions indicated by mode. This computation is batched and expects a 2D array
    for x and a 3D array for Y.

    Assumes points to be ordered properly (e.g. time).
    From: https://arxiv.org/abs/1705.08051

    Args:
        x (np.ndarray): Sequence of points
        Y (np.ndarray): List of sequences of points
        max_value (np.ndarray): max for all dimensions
        x_mask (np.ndarray): Mask for padding in x
        Y_mask (np.ndarray): Mask for padding in Y
        mode (str): Mode of the counting distance on sequences ordered by time. Can be
            - 'counting_tpp': for counting distance on time
            - 'counting_stpp': for counting distance on space and time
            - 'counting_spp': for counting distance on space

    Returns:
        np.ndarray: distance
    """

    # All of the following functions assume that the sequences are ordered by time, but
    # then only TPP, SPP or STPP part is used for the distance computation.
    if mode == "counting_tpp":
        # This assumes time to be the first dimension.
        # Note that this also works for other 1D spaces than time.
        x, Y = x.copy()[:, :1], Y.copy()[:, :, :1]
        x_mask, Y_mask = x_mask.copy()[:, :1], Y_mask.copy()[:, :, :1]
        max_value = max_value.copy()[:1]
    elif mode == "counting_spp":
        # Assumes all dimension except the first one to be space.
        # Note that this would also not be limited to just time
        x, Y = x.copy()[:, 1:], Y.copy()[:, :, 1:]
        x_mask, Y_mask = x_mask.copy()[:, 1:], Y_mask.copy()[:, :, 1:]
        max_value = max_value.copy()[1:]
    elif mode == "counting_stpp":
        x, Y = x.copy(), Y.copy()
        x_mask, Y_mask = x_mask.copy(), Y_mask.copy()
        max_value = max_value.copy()

    b, s, dim = Y.shape

    # Repeat data to match batch size
    x = repeat(x, "s dim -> b s dim", b=b)
    x_mask = repeat(x_mask, "s dim -> b s dim", b=b)
    max_value = repeat(max_value, "dim -> b s dim", b=b, s=s)

    # Ensure that the the first sequence is larger than the second
    x_len = reduce(x_mask, "b s dim -> b", "sum") / dim
    y_len = reduce(Y_mask, "b s dim -> b", "sum") / dim
    to_swap = x_len > y_len

    # Swap sequences
    x[to_swap], Y[to_swap] = Y[to_swap].copy(), x[to_swap].copy()
    x_mask[to_swap], Y_mask[to_swap] = (
        Y_mask[to_swap].copy(),
        x_mask[to_swap].copy(),
    )

    # Compute the distance
    result = reduce(np.abs(x - Y) * x_mask, "b s dim -> b", "sum")
    result += reduce((max_value - Y) * (~x_mask & Y_mask), "b s dim -> b", "sum")

    # Normalize for number of dimensions
    return result / dim


@typechecked
def mae_length_forecast_distance(  # MAE error
    X: List[np.ndarray],
    Y: List[np.ndarray],
):
    """
    Computes the MAE of the length of x and the forecasted Y.

    Args:
        X (np.ndarray): List of sequences
        Y (np.ndarray): List of sequences

    Returns:
        mae (float): Distance
    """
    # Get length of sequences.
    X_lengths = np.array([s.shape[0] for s in X])
    Y_lengths = np.array([s.shape[0] for s in Y])
    mae = np.mean(np.abs(X_lengths - Y_lengths))
    return mae


@typechecked
def wasserstein_point_distance(
    x: np.ndarray,
    Y: List[np.ndarray],
):
    """
    Computes the Wasserstein distance of the distribution of the location of points.

    Args:
        x (np.ndarray): Event times for x
        Y (List[np.ndarray]): List of sequences

    Returns:
        np.ndarray: distance
    """
    distance = []
    for y in Y:
        if len(y) > 0:
            distance.append(sliced_wasserstein_distance(x, y))
    return distance


@typechecked
def points_distribution_wasserstein_distance(
    X: List[np.ndarray],
    Y: List[np.ndarray],
):
    """
    Returns the Wasserstein between the distribution of points between sequences X and Y.

    Args:
        X (List): List of sequences
        Y (List): List of sequences

    Returns:
        float: Wasserstein distance
    """
    # Get all points as a sequence
    return sliced_wasserstein_distance(np.concatenate(X), np.concatenate(Y))


@typechecked
def projection_distance(
    x: np.ndarray,
    Y: np.ndarray,
    x_mask: np.ndarray,
    Y_mask: np.ndarray,
    max_value: np.ndarray,
    dim: int,
    n_projections: int = 50,
    seed: int = 80672983,
):
    """
    Computes the distance between the projection of the points onto the identity line in the given
    dimensional space.

    Assumes points to be in [-1,1].

    Args:
        x (np.ndarray): Event times for x
        Y (np.ndarray): List of sequences
        dim (int): Dimensions of the space
        n_projections (int): Number of random projections to compute
        seed (int): Seed for random line generation

    Returns:
        np.ndarray: distance
        np.ndarray: normalized distance

    """
    # Create random seed generator
    rng = np.random.default_rng(seed)

    # Matrix to store all results for all projections
    results = np.zeros((len(Y), n_projections))

    for i in range(n_projections):
        # Project the points onto a random line in the space
        proj_line = rng.normal(
            loc=0,
            scale=1.0,
            size=dim,
        )[..., None]
        proj_line = proj_line / np.linalg.norm(proj_line)

        # Project points and make them one dimensional
        x_proj = np.matmul(x, proj_line)
        Y_proj = np.matmul(Y, proj_line)

        max_proj = np.array([np.abs(proj_line).sum()])

        x_proj = x_proj / max_proj
        Y_proj = Y_proj / max_proj

        # assert (x_proj <= 1).all() and (x_proj >= -1).all(), "x_proj in [-1,1]"
        # assert (Y_proj <= 1).all() and (Y_proj >= -1).all(), "Y_proj in [-1,1]"

        # Order increasingly every sequence
        sorted_ind_x = np.argsort(x_proj, axis=0)
        x_proj = np.take_along_axis(x_proj, sorted_ind_x, axis=0)
        mask_x = np.take_along_axis(x_mask[:, :1], sorted_ind_x, axis=0)

        sorted_ind_Y = np.argsort(Y_proj, axis=1)
        Y_proj = np.take_along_axis(Y_proj, sorted_ind_Y, axis=1)
        mask_y = np.take_along_axis(Y_mask[:, :, :1], sorted_ind_Y, axis=1)

        # Compute 1D counting distances between projected x and Y.
        result = counting_distance(
            x_proj, Y_proj, mask_x, mask_y, np.array([1]), "counting_tpp"
        )

        # Save distances on this projection line
        results[:, i] = result

    # For every sequence compute mean over all projections
    avg_result = results.mean(axis=1)
    return avg_result


@typechecked
def lengths_distribution_wasserstein_distance(
    X: List[np.ndarray], Y: List[np.ndarray], mean_number_items: float
):
    """
    Returns the Wasserstein between the distribution of sequence lengths between X and Y.

    Args:
        X (List): List of sequences
        Y (List): List of sequences
        mean_number_items (float): Mean number of events from the dataset. This is used for normalization

    Returns:
        float: Wasserstein distance
    """
    # Get length of sequences
    X_lengths = np.array([s.shape[0] for s in X])
    Y_lengths = np.array([s.shape[0] for s in Y])
    return wasserstein_distance(
        X_lengths / mean_number_items, Y_lengths / mean_number_items
    )


@typechecked
def MMD(
    X: List,
    Y: List,
    distance: str,
    dim: int,
    space_bound: np.ndarray,
    sample_size: int = None,
    sigma: float = None,
    seed: int = 9012835,
):
    """
    Computes the maximum mean discrepency between the samples X and samples Y.
    MMD is defined as E[k(x, x)] - 2*E[k(x, y)] + E[k(y, y)]. We use a Gaussian kernel
    with the counting distance. k(x, y) = exp(-d(x, y)/(2*sigma2)) where d is a given distance
    and sigma is either given or estimated as the median distance between all pairs.

    Args:
        X (List): List of sequences
        Y (List): List of sequences
        distance (str): Distance to compute in the MMD. Can be
            - 'wasserstein': for 2D wasserstein distance on space
            - 'projection': for self-developed projection distance
            - 'counting_tpp': for counting distance on time
            - 'counting_stpp': for counting distance with L1 distance on space and time
            - 'counting_spp': for counting distance with L1 distance on space ordered by time
        dim (int): dimension of the bounded space
        space_bound (np.ndarray): space boundaries of original dataset
        sample_size (int, optional): If given MMD is only computed for subsets of X and Y.
            This improves performance at the cost of performance. Defaults to None.
        sample_perm (int, 10 by default): Number of permutations to check the sequence distance minimum.
        sigma (float, optional): Sigma for the Gaussian kernel. If not given it is estimated. Defaults to None.

    Returns:
        float: MMD(X, Y)
    """
    assert distance in (
        "counting_tpp",
        "counting_stpp",
        "counting_spp",
        "wasserstein",
        "projection",
    ), f"Distance {distance} not implemented"

    # Subsample samples from distributions
    if sample_size is not None:
        if sample_size < len(X):
            X = [X[i] for i in np.random.choice(len(X), sample_size, replace=False)]
        if sample_size < len(Y):
            Y = [Y[i] for i in np.random.choice(len(Y), sample_size, replace=False)]

    # Do reshape in case of projection distance
    if distance != "wasserstein":
        X, Y, X_mask, Y_mask = match_shapes(X, Y, dim)
    else:
        X_mask = None
        Y_mask = None

    def distance_n_2_n(X, Y, X_mask, Y_mask, max_value, dim, seed, distance):
        x_y_d = []
        for i in range(len(X)):
            if distance == "wasserstein":
                if len(X[i]) > 0:
                    x_y_d.append(wasserstein_point_distance(X[i], Y))
            elif distance == "projection":
                x_y_d.append(
                    projection_distance(
                        x=X[i],
                        Y=Y,
                        x_mask=X_mask[i],
                        Y_mask=Y_mask,
                        max_value=max_value,
                        dim=dim,
                        seed=seed,
                    )
                )
            else:
                x_y_d.append(
                    counting_distance(
                        x=X[i],
                        Y=Y,
                        x_mask=X_mask[i],
                        Y_mask=Y_mask,
                        max_value=max_value,
                        mode=distance,
                    )
                )
        x_y_d = np.concatenate(x_y_d)
        return x_y_d

    x_x_d = distance_n_2_n(
        X,
        X,
        X_mask,
        X_mask,
        space_bound[:, 1],
        dim=dim,
        seed=seed,
        distance=distance,
    )
    y_y_d = distance_n_2_n(
        Y,
        Y,
        Y_mask,
        Y_mask,
        space_bound[:, 1],
        dim=dim,
        seed=seed,
        distance=distance,
    )
    x_y_d = distance_n_2_n(
        X,
        Y,
        X_mask,
        Y_mask,
        space_bound[:, 1],
        dim=dim,
        seed=seed,
        distance=distance,
    )

    if sigma is None:
        sigma = np.median(np.concatenate([x_x_d, x_y_d, y_y_d]))

    sigma2 = sigma**2
    E_x_x = np.mean(gaussian_kernel(x_x_d, sigma2))
    E_x_y = np.mean(gaussian_kernel(x_y_d, sigma2))
    E_y_y = np.mean(gaussian_kernel(y_y_d, sigma2))

    return np.sqrt(E_x_x - 2 * E_x_y + E_y_y), sigma
