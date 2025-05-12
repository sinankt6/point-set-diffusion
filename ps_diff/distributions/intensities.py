from typing import Tuple

import torch
import torch.distributions as D
import torch.nn as nn
import warnings
from torch.distributions import MixtureSameFamily
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from ps_diff.data import Batch
from ps_diff.distributions.densities import DISTRIBUTIONS

patch_typeguard()


@typechecked
class MixtureIntensity(nn.Module):
    """
    Class parameterizing the intensity function as a weighted mixture of distributions.

    Parameters:
    ----------
    dim: int,
        Dimensionality of SPP points
    n_components : int, optional
        Number of components to use in the mixture, by default 10
    embedding_size : int, optional
        Size of the event embedding, by default 128
    distribution : str, optional
        Distribution to use for the components, by default "multivar_normal"

    """

    def __init__(
        self,
        dim: int,
        n_components: int = 10,
        embedding_size: int = 64,
        distribution: str = "multivar_normal",
    ) -> None:
        super().__init__()

        assert (
            distribution in DISTRIBUTIONS.keys()
        ), f"{distribution} not in {DISTRIBUTIONS.keys()}"
        self.w_activation = torch.nn.Softplus()
        self.distribution = DISTRIBUTIONS[distribution]

        # Get dimensions of bounded space
        self.dim = dim

        # Parallel compute parameters weight for n components,
        # mu and sigma of d dimensions for n components dimensions with one MLP
        self.n_components = n_components
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_size, 2 * embedding_size),
            nn.ReLU(),
            nn.Linear(2 * embedding_size, (2 * dim + 1) * n_components),
        )
        # Kaiming initialization for weights
        self._init_mlp_weights()

        self.rejections_sample_multiple = 2

    def _init_mlp_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_intensity_parameters(
        self,
        x_n: Batch,
        event_emb: TensorType[float, "batch", "seq", "embedding"],
        dif_time_emb: TensorType[float, "batch", "embedding"],
    ) -> Tuple[TensorType, TensorType, TensorType]:
        """
        Compute the parameters of the intensity function.

        Parameters:
        ----------
        x_n : Batch
            Batch of event sequences to condition on
        event_emb : TensorType[float, "batch", "seq", "embedding"]
            Context embedding of the events
        dif_time_emb : TensorType[float, "batch", "embedding"]
            Embedding of the diffusion time

        Returns:
        -------
        location, scale, weight: List[TensorType]
            The parameters of the intensity function
        """

        # Compute masked mean over sequence (zero padded)
        n_events = x_n.mask.sum(1)[:, 0]
        seq_emb = event_emb.sum(1) / torch.clamp(n_events[..., None], min=1)

        # Location and covariance have n components on d dimensions. Weights have n components
        parameters = self.mlp(torch.cat([seq_emb, dif_time_emb], dim=-1))
        return torch.split(
            parameters,
            [
                self.n_components * self.dim,
                self.n_components * self.dim,
                self.n_components,
            ],
            dim=-1,
        )

    def get_distribution(
        self,
        event_emb: TensorType[float, "batch", "seq", "embedding"],
        dif_time_emb: TensorType[float, "batch", "embedding"],
        x_n: Batch,
        L,
    ) -> Tuple[D.MixtureSameFamily, TensorType[float, "batch"]]:
        """
        Instantiate the mixture-distribution parameterizing the intensity function.

        Parameters:
        ----------
        event_emb : TensorType[float, "batch", "seq", "embedding"]
            Context embedding of the events
        dif_time_emb : TensorType[float, "batch", "embedding"]
            Embedding of the diffusion time
        x_n : Batch
            Batch of event sequences to condition on
        L : int
            Maximum sequence length

        Returns:
        -------
        density, cumulative_intensity: Tuple[D.MixtureSameFamily, TensorType[float, "batch"]]
            The distribution and the cumulative intensity
        """
        location, scale, weight = self.get_intensity_parameters(
            x_n=x_n,
            event_emb=event_emb,
            dif_time_emb=dif_time_emb,
        )
        # Get batch shape
        B, _ = location.shape

        # Reshape the loc, scale intensity parameters to [n_components, dim]
        location = location.reshape(B, self.n_components, self.dim)
        scale = scale.reshape(B, self.n_components, self.dim)

        # Include the number of events in x_n for the cumulative intensity
        weight = self.w_activation(weight)
        cumulative_intensity = (weight).sum(-1) * (x_n.mask.sum(1)[:, 0] + 1)

        # Probs is normalized to sum to 1
        mixture_dist = D.Categorical(
            probs=weight.unsqueeze(1).repeat(1, L, 1),
            validate_args=False,  # avoid bug on simplex constraint
        )

        # Distribution parameters are the same for each sequence element
        component_dist = self.distribution(
            location.unsqueeze(1).repeat(1, L, 1, 1),
            scale.unsqueeze(1).repeat(1, L, 1, 1),
        )
        return (
            MixtureSameFamily(mixture_dist, component_dist),
            cumulative_intensity,
        )

    def log_likelihood(
        self,
        x_0: Batch,
        event_emb: TensorType[float, "batch", "seq", "embedding"],
        dif_time_emb: TensorType[float, "batch", "embedding"],
        x_n: Batch,
    ) -> TensorType[float, "batch"]:
        """
        Compute the log-likelihood of the event sequences.

        Parameters:
        ----------
        x_0 : Batch
            Batch of event sequences
        event_emb : TensorType[float, "batch", "seq", "embedding"]
            Context embedding of the events
        dif_time_emb : TensorType[float, "batch", "embedding"]
            Embedding of the diffusion time
        x_n : Batch
            Batch of event sequences to condition on

        Returns:
        -------
        log_likelihood: TensorType[float, "batch"]
            The log-likelihood of the event sequences
        """
        density, cif = self.get_distribution(
            event_emb=event_emb,
            dif_time_emb=dif_time_emb,
            x_n=x_n,
            L=x_0.seq_len,
        )

        # Compute log-intensity with re-weighting
        log_intensity = (
            (density.log_prob(x_0.points) + torch.log(cif)[..., None])
            * x_0.mask[:, :, 0]
        ).sum(-1)

        # Compute CIF for normalization
        cdf = density.cdf(torch.ones_like(x_0.points)).mean(1)
        cif = cif * cdf  # Rescale to original shape

        return log_intensity - cif

    def sample(
        self,
        event_emb: TensorType[float, "batch", "seq", "embedding"],
        dif_time_emb: TensorType[float, "batch", "embedding"],
        n_samples: int,
        x_n: Batch,
    ) -> Batch:
        """
        Sample event sequences from the intensity function.

        Parameters:
        ----------
        event_emb : TensorType[float, "batch", "seq", "embedding"]
            Context embedding of the events
        dif_time_emb : TensorType[float, "batch", "embedding"]
            Embedding of the diffusion time
        n_samples : int
            Number of samples to draw
        x_n : Batch
            Batch of event sequences to condition on

        Returns:
        -------
        Batch
            The sampled event sequences
        """
        # Get normalized boundaries
        norm_space_bound = torch.cat(
            (
                torch.ones(x_n.dim)[..., None] * (-1),
                torch.ones(x_n.dim)[..., None] * (1),
            ),
            dim=-1,
        ).to(x_n.points.device)

        l_norm_bound = norm_space_bound[:, 0]
        u_norm_bound = norm_space_bound[:, 1]

        density, cif = self.get_distribution(
            event_emb=event_emb,
            dif_time_emb=dif_time_emb,
            x_n=x_n,
            L=1,
        )

        sequence_len = torch.round(
            cif
            * density.cdf(torch.ones(n_samples, 1, device=event_emb.device)).squeeze()
        ).long()

        # TODO implement smarter truncated normal, without rejection sampling.
        max_seq_len = sequence_len.max()

        while True:
            points = (
                density.sample(((max_seq_len + 1) * self.rejections_sample_multiple,))
                .squeeze(2)
                .permute(1, 0, 2)
            )

            # Reject if not in space_bound
            inside = (
                torch.logical_and(points <= u_norm_bound, points >= l_norm_bound).sum(
                    -1
                )
                == self.dim
            )  # Check points inside every dimensional boundary
            sort_idx = torch.argsort(inside.int(), stable=True, descending=True, dim=-1)
            inside = torch.take_along_dim(inside, sort_idx, dim=1)[:, :max_seq_len][
                ..., None
            ].repeat(
                1, 1, self.dim
            )  # Expand indices for all dimensions of space
            points = torch.take_along_dim(
                points, sort_idx[..., None].repeat(1, 1, self.dim), dim=1
            )[:, :max_seq_len, :]
            # Randomly mask out events exceeding the actual sequence length
            mask = (
                torch.arange(0, points.shape[1], device=points.device)[None, :]
                < sequence_len[:, None]
            )[..., None].repeat(
                1, 1, self.dim
            )  # expand mask bool along all dimensions
            mask = mask * inside

            if (mask[:, :, 0].sum(1) == sequence_len).all():
                break
            else:
                self.rejections_sample_multiple += 1
                warnings.warn(
                    f"""
Rejection sampling multiple increased to {self.rejections_sample_multiple}, as not enough event points were inside space_bound.
""".strip()
                )

        points = points * mask

        return Batch.remove_unnescessary_padding(
            points=points, mask=mask, space_bound=x_n.space_bound, kept=None
        )
