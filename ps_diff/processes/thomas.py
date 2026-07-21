from typing import Union, List

import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from ps_diff.data import Batch

patch_typeguard()  # use before @typechecked


@typechecked
def generate_thomas(
    space_bound: TensorType[float, "dim", 2],
    n_sequences: int,
    cluster_dims: List[int],
    parent_intensity: float,
    offspring_intensity: Union[TensorType, None] = None,
    offspring_scale: float = 1.0,
    cluster_std: Union[TensorType, None]  = None,
    buffer_sigma: float = 4.0,
) -> Batch:
    """
    Generate a batch of sequences from a Thomas cluster process on the bounded space S.

    Parameters
    ----------
    space_bound : TensorType[float, "dim", 2]
        Boundaries of the metric space. If the metric space is in R^d consists of lower
        and upper bound of each dimension with [dim, 2] shape
    n_sequences : int
        Number of sequences to generate
    cluster_dims : List[int]
        Indices of the dimensions that should be clustered via the Thomas process. For only spatial clustering, pass [1, 2]. [0] (Time) stays homogenous.
    parent_intensity : float
        Expected number of clusters per unit Area
    offspring_intensity : Union[TensorType, None], optional
        per sequence scaling of the offspring rate, set to 1 if None, by default None
    offspring_scale : float, optional
        Global scale for the offspring rate, 1.0 by default
    cluster_std : Union[TensorType, None] = None,
        Std of the offspring jittering around each parent. One vlaue per entry in cluster_dims (anisotropic). Set to 1 for all dims if None.
    buffer_sigma : float, optional
        Parent sampling window extension by buffersigma*cluster_std per dim to reduce edge effects, 4.0 by default.

    Returns
    -------
    Batch
        Batch of generated sequences
    """
    device = space_bound.device

    # Get dimensionality of bounded space
    dim = space_bound.shape[0]
    #set homogenouns dimensions
    homog_dims = [d for d in range(dim) if d not in cluster_dims]
    #count number of cluster dimensions
    n_cluster_dims = len(cluster_dims)

    # like in hpp with intensity
    if offspring_intensity is None: 
        offspring_intensity = torch.ones(n_sequences, device=device)
    if cluster_std is None:
        cluster_std = torch.ones(n_cluster_dims, device=device)

    #same width calculationa as in hpp just for our cluster_dims
    space_bound_cluster = space_bound[cluster_dims]
    width_cluster = space_bound_cluster[:, 1] - space_bound_cluster[:, 0]

    #parent process ----------------------------------------
    # buffer for the edges
    buffer = buffer_sigma * cluster_std
    buf_width = width_cluster + 2 * buffer
    buf_vol = buf_width.prod() # area of the buffered zone

    # Get number of samples and choose maximum number to later generate
    # all samples
    n_parents = torch.poisson(
        buf_vol * parent_intensity * torch.ones(n_sequences, device=device)
    )
    # für den tensor das der max anzahl weiß (an sich aber poisson zufällig)
    max_parents = int(n_parents.max().item()) + 1

    #gleichverteilte zahlen zwischen 0 und 1 ziehen und linear auf grenzen mit puffer reskalieren
    parent_positions = (
        torch.rand((n_sequences, max_parents, n_cluster_dims), device=device)
        * buf_width[None, None, :]
        + (space_bound_cluster[:, 0] - buffer)[None, None, :]
    )
    #wie viele sind echt, wie viele sind nur Padding
    parent_mask = (
        torch.arange(max_parents, device=device)[None, :] < n_parents[: None]
    )

    #child process ------------------------------------------
    #fake parents (pading) receive 0 childs because of mask
    offspring_rate_per_parent = (offspring_intensity * offspring_scale)[: None] * parent_mask.float()
    n_offspring = torch.poisson(mu) #each parent receives number of children
    max_offspring_per_parent = int(n_offspring.max().item()) + 1


    offsets = torch.randn(
        (n_sequences, max_parents, max_offspring_per_parent, n_cluster_dims), device=device
    ) * cluster_std[None, None, None, :]

    offspring_positions = parent_positions[:, :, None, :] + offsets

    #mask to check if padding or real child 
    offspring_valid = (
        torch.arange(max_offspring_per_parent, device=device)[None, None, :]
        < n_offspring[:, :, None]
    )
    #check if points are in not buffered space
    in_domain = (
        (offspring_positions >= space_bound_cluster[:, 0]).all(-1)
        & (offspring_positions <= space_bound_cluster[:, 1]).all(-1)
    )
    valid = offspring_valid & in_domain

    #count number of valid points per sequence
    flat_positions = offspring_positions.reshape(n_sequences, max_parents * max_offspring_per_parent, n_cluster_dims)
    flat_valid = valid.reshape(n_sequences, max_parents * max_offspring_per_parent)

    n_samples = flat_valid.sum(dim=1)
    max_samples = int(n_samples.max().item()) + 1

    #sortieren, sodass gültige punkte vorne rest hinten
    sort_idx = torch.argsort((~flat_valid).float(), dim=1, stable=True)
    flat_positions_sorted = torch.gather(
        flat_positions, 1, sort_idx[..., None].expand(-1, -1, n_cluster_dims)
    )[:, :max_samples]

    mask_1d = torch.arange(max_samples, device=device)[None, :] < n_samples[:, None]
    mask = mask_1d[..., None].repeat(1, 1, n_cluster_dims)
    cluster_points = mask * flat_positions_sorted

    #from hpp
    assert (mask.sum(1)[:, 0] == n_samples).all(), "wrong number of cluster samples"

    #add time dimension
    if homog_dims: 
        space_bound_homog = space_bound[homog_dims]
        width_homog = space_bound[:, 1] - space_bound_homog[:, 0]
        #uniform sampling
        homog_points = (
            torch.rand((n_sequences, max_samples, len(homog_dims)), device=device)
            *width_homog[None, None, :]
            * space_bound_homog[:, 0][None, None, :]
        )
        homog_points = homog_points * mask_1d[..., None]

        points = torch.zeros((n_sequences, max_samples, dim), device=device)
        points[..., cluster_dims] = cluster_points
        points[..., homog_dims] = homog_points
    else:
            points = cluster_points

    full_mask = mask_1d[..., None].repeat(1, 1, dim)

    return Batch.remove_unnescessary_padding(
        points=points, mask=full_mask, space_bound=space_bound, kept=None
    )