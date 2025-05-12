from typing import Union

import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from ps_diff.data import Batch

patch_typeguard()  # use before @typechecked


@typechecked
def generate_hpp(
    space_bound: TensorType[float, "dim", 2],
    n_sequences: int,
    intensity: Union[TensorType, None] = None,
    scale: float = 1.0,
) -> Batch:
    """
    Generate a batch of sequences from a homogeneous Poisson process on the bounded space S.

    Parameters
    ----------
    space_bound : TensorType[float, "dim", 2]
        Boundaries of the metric space. If the metric space is in R^d consists of lower
        and upper bound of each dimension with [dim, 2] shape
    n_sequences : int
        Number of sequences to generate
    intensity : Union[TensorType, None], optional
        Intensity of the process, set to 1 if None, by default None
    scale : float, optional
        Scaling for the intensity, 1.0 by default

    Returns
    -------
    Batch
        Batch of generated sequences
    """
    device = space_bound.device

    # Get dimensionality of bounded space
    dim = space_bound.shape[0]

    if intensity is None:
        intensity = torch.ones(n_sequences, device=device)

    # Compute the width of each dimension
    # width: [n_seq, dim]
    width = space_bound[:, 1] - space_bound[:, 0]
    # Compute the volume of the bounded spaces
    # vol: [n_seq, dim]
    vol = width.prod()

    # Get number of samples and choose maximum number to later generate
    # all samples
    n_samples = torch.poisson(vol * intensity * scale)
    max_samples = int(torch.max(n_samples).item()) + 1

    # Sample points using broadcasting
    # points: [n_seq, max_samples, dim]
    points = (
        torch.rand((n_sequences, max_samples, dim), device=device)
        * width[None, None, ...]
        + space_bound[:, 0][None, None, ...]
    )

    # Mask for padding events[]
    # mask: [n_seq, max_samples, dim]
    mask = (
        torch.arange(0, max_samples, device=device)[None, :]
        < n_samples[:, None]
    )
    mask = mask[..., None].repeat(1, 1, dim)
    points = mask * points

    # Check correct number of samples after padding
    assert (mask.sum(1)[:, 0] == n_samples).all(), "wrong number of samples"

    return Batch.remove_unnescessary_padding(
        points=points, mask=mask, space_bound=space_bound, kept=None
    )


@typechecked
def rescale_normhpp(
    original_space_bound: TensorType[float, "dim", 2],
    n_sequences: int,
    intensity: Union[TensorType, None] = None,
    scale: float = 1.0,
) -> Batch:
    """
    Generate a batch of sequences from a homogeneous Poisson process on a normalized bounded
    space of volume 2^dim: S = [-1, 1]^dim

    Parameters
    ----------
    original_space_bound : TensorType[float, "dim", 2]
        Boundaries of the metric space. If the metric space is in R^d consists of lower
        and upper bound of each dimension with [dim, 2] shape
    n_sequences : int
        Number of sequences to generate
    intensity : Union[TensorType, None], optional
        Intensity of the process, set to 1 if None, by default None
    scale : float, optional
        Scaling for the intensity, 1.0 by default

    Returns
    -------
    Batch
        Batch of generated sequences
    """
    device = original_space_bound.device

    # Get dimensionality of bounded space
    dim = original_space_bound.shape[0]

    # Get normalized space bound [-1.0, 1.0]^dim
    norm_bounded_space = torch.cat(
        (
            -torch.ones(dim)[..., None],
            torch.ones(dim)[..., None],
        ),
        dim=-1,
    ).to(device)

    # Generate normalized HPP on bounded space of volume one
    batch = generate_hpp(
        space_bound=norm_bounded_space,
        n_sequences=n_sequences,
        intensity=intensity,
        scale=scale,
    )

    batch.space_bound = original_space_bound

    return batch
