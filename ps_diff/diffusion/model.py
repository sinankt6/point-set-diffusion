import math
import torch
import torch.nn as nn

from typing import Tuple
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from ps_diff.data import Batch
from ps_diff.backbones.attention import AttentionPointEmb
from ps_diff.backbones.embeddings import NyquistFrequencyEmbedding
from ps_diff.processes.hpp import rescale_normhpp
from ps_diff.processes.thomas import rescale_norm_thomas 
from ps_diff.diffusion.utils import thin_and_add_probs


patch_typeguard()


@typechecked
class PSDiff(nn.Module):
    """
    Implementation of Point Set Diffusion.

    Parameters
    ----------
    classifier_model : nn.Module
        Model for predicting the intersection of x_0 and x_n from x_n
    intensity_model : nn.Module
        Model for predicting the intensity of x_0 \ x_n
    space_bound : TensorType[float, "dim", 2]
        Boundaries of the space of spatial point processes
    n_max : int, optional
        Maximum number of events, by default 100
    steps : int, optional
        Number of diffusion steps, by default 100
    hpp_scale : float, optional
        Scaling for the HPP process, by default 1.0
    emb_dim : int, optional
        Embedding dimensions of the models, by default 64
    encoder_n_block : int, optional
        Number of encoder layers, by default 4
    """

    def __init__(
        self,
        classifier_model,
        intensity_model,
        space_bound: TensorType[float, "batch", 2],
        n_max: int = 100,
        steps: int = 100,
        hpp_scale: float = 1.0,
        emb_dim: int = 64,
        encoder_n_blocks: int = 4,
        max_time=100,
        noise_process: str = "hpp",
        thomas_kappa: float = None,
        thomas_mu: float = None,
        thomas_cluster_std: TensorType = None,
        thomas_cluster_dims: list = None,  
    ) -> None:
        super().__init__()
        self.steps = steps

        retain, add_prob, add_keep, retain_x0_kept = thin_and_add_probs(
            torch.tensor(
                [
                    math.cos((n / steps) / 1 * math.pi / 2) ** 2
                    for n in range(steps)
                ],
                dtype=torch.float64,
            )
        )

        self.register_buffer("retain", retain)
        self.register_buffer("add_prob", add_prob)
        self.register_buffer("add_keep", add_keep)
        self.register_buffer("retain_x0_kept", retain_x0_kept)

        # Set models for the approximate posterior
        self.classifier_model = classifier_model
        self.intensity_model = intensity_model

        self.dim = space_bound.shape[0]
        self.space_bound = space_bound

        self.n_max = n_max
        self.max_time = max_time
        self.hpp_scale = hpp_scale

        self.noise_process = noise_process
        self.thomas_kappa = thomas_kappa
        self.thomas_mu = thomas_mu
        self.thomas_cluster_dims = thomas_cluster_dims
        if thomas_cluster_std is not None:
            self.register_buffer("thomas_cluster_std", thomas_cluster_std)
        else:
            self.thomas_cluster_std = None

        self.set_encoders(
            emb_dim=emb_dim,
            encoder_n_blocks=encoder_n_blocks,
            steps=steps,
        )

    def _sample_noise(
        self,
        space_bound: TensorType[float, "dim", 2],
        n_sequences: int,
        intensity: TensorType = None,
    ) -> Batch:
        if self.noise_process == "hpp":
            return rescale_normhpp(
                original_space_bound=space_bound,
                n_sequences=n_sequences,
                intensity=intensity,
                scale=self.hpp_scale,
            )
        elif self.noise_process == "thomas":
            return rescale_norm_thomas(
                original_space_bound=space_bound,
                n_sequences=n_sequences,
                cluster_dims=self.thomas_cluster_dims,
                parent_intensity=self.thomas_kappa,
                intensity=intensity,
                scale=self.thomas_mu,
                cluster_std=self.thomas_cluster_std,
            )
        else:
            raise ValueError(f"Unknown noise_process: {self.noise_process}")

    def set_encoders(
        self,
        emb_dim: int,
        encoder_n_blocks: int,
        steps: int,
    ) -> None:
        """
        Set the encoders for the model.

        Parameters
        ----------
        emb_dim : int
            Embedding dimensions of the model
        encoder_n_blocks : int
            Number of encoder blocks
        steps : int
            Number of diffusion steps
        """

        # Diffusion time encoder
        position_diff_emb = NyquistFrequencyEmbedding(
            dim=emb_dim, timesteps=steps
        )
        self.diffusion_time_encoder = nn.Sequential(
            position_diff_emb,
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
        # Initialize weights in encoder
        for m in self.diffusion_time_encoder:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Event sequence encoder
        self.sequence_encoder = AttentionPointEmb(
            n_blocks=encoder_n_blocks,
            input_dim=self.dim,
            embs_dim=emb_dim // 2,
        )

        position_len_emb = NyquistFrequencyEmbedding(
            dim=emb_dim // 2, timesteps=self.n_max
        )
        self.seq_len_encoder = nn.Sequential(
            position_len_emb,
            nn.Linear(emb_dim // 2, emb_dim // 2),
            nn.GELU(),
            nn.Linear(emb_dim // 2, emb_dim // 2),
        )
        # Initialize weights in encoder
        for m in self.seq_len_encoder:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def compute_emb(
        self, n: TensorType[torch.long, "batch"], x_n: Batch
    ) -> Tuple[
        TensorType["batch", "embedding"],
        TensorType["batch", "sequence", "embedding"],
    ]:
        """
        Get the embeddings of x_n.

        Parameters
        ----------
        n : TensorType[torch.long, "batch"]
            Diffusion time step
        x_n : Batch
            Batch of data

        Returns
        -------
        Tuple[
            TensorType["batch", "embedding"],
            TensorType["batch", "sequence", "embedding"],
        ]
            Diffusion time embedding, event sequence embedding
        """

        # Embed diffusion time
        dif_time_emb = self.diffusion_time_encoder(n)

        # Get point embedding
        point_emb = self.sequence_encoder(x_n)
        # Get length sequence embedding
        len_emb = self.seq_len_encoder(x_n.unpadded_length)
        len_emb = len_emb.unsqueeze(1).repeat(1, point_emb.shape[1], 1)

        event_emb = (
            torch.cat([point_emb, len_emb], dim=-1)
            * x_n.mask[:, :, 0][..., None]
        )

        return (
            dif_time_emb,
            event_emb,
        )

    def get_n(self, shape, device, min=None, max=None) -> TensorType[int]:
        """
        Uniformly sample n, i.e., the diffusion time step per sequence.

        Parameters
        ----------
        shape :
            Shape of the tensor
        device :
            Device of the tensor
        min : None, optional
            Minimum value of n, by default None
        max : None, optional
            Maximum value of n, by default None

        Returns
        -------
        TensorType[int]
            Sampled n
        """
        if min is None or max is None:
            min = 0
            max = self.steps
        return torch.randint(
            min,
            max,
            size=shape,
            device=device,
            dtype=torch.long,
        )

    def noise(
        self, x_0: Batch, n: TensorType[torch.long, "batch"]
    ) -> Tuple[Batch, Batch]:
        """
        Sample x_n from x_0 by applying the noising process.

        Parameters
        ----------
        x_0 : Batch
            Batch of data
        n : TensorType[torch.long, "batch"]
            Number of noise steps

        Returns
        -------
        Tuple[Batch, Batch]
            x_n and thinned x_0
        """
        # Thin x_0
        x_0_kept, x_0_thinned = x_0.thin(alpha=self.retain[n])

        # Superposition with HPP (add) or Thomas
        noise_events = self._sample_noise(
            space_bound=x_0.space_bound,
            n_sequences=len(x_0),
            intensity=self.add_prob[n],
        )

        x_n = x_0_kept.add_events(noise_events)

        return x_n, x_0_thinned

    def forward(self, x_0: Batch) -> Tuple[
        TensorType[float, "batch", "sequence_x_n"],
        TensorType[float, "batch"],
        Batch,
    ]:
        """
        Forward pass to train the model, i.e., predict x_0 from x_n.

        Parameters
        ----------
        x_0 : Batch
            Batch of data

        Returns
        -------
        Tuple[
            TensorType[float, "batch", "sequence_x_n"],
            TensorType[float, "batch"],
            Batch,
        ]
            classification logits, log likelihood of x_0 without x_n, noised data
        """
        # Uniformly sample n
        n = self.get_n(
            min=0,
            max=self.steps,
            shape=(len(x_0),),
            device=x_0.points.device,
        )
        # Noise x_0 to get x_n
        x_n, x_0_thin = self.noise(x_0=x_0, n=n)
        # Get embeddings
        (dif_time_emb, event_emb) = self.compute_emb(n=n, x_n=x_n)

        # Predict intersection x_0 and x_n from x_n
        x_n_and_x_0_logits = self.classifier_model(
            dif_time_emb=dif_time_emb,
            event_emb=event_emb,
        )

        # Evaluate intensity of thinned x_0
        log_like_x_0 = self.intensity_model.log_likelihood(
            event_emb=event_emb,
            dif_time_emb=dif_time_emb,
            x_0=x_0_thin,
            x_n=x_n,
        )

        return x_n_and_x_0_logits, log_like_x_0, x_n

    def sample(
        self, n_samples: int, space_bound: TensorType[float, "dim", 2]
    ) -> Batch:
        """
        Sample x_0 starting from x_N.

        Parameters
        ----------
        n_samples : int
            Number of samples
        space_bound : TensorType[float, "dim", 2]
            Boundaries of the space of spatial point processes

        Returns
        -------
        Batch
            Sampled x_0s
        """

        # Init x_N by sampling from HPP
        x_N = self._sample_noise(
            space_bound=space_bound,
            n_sequences=n_samples,
        )
        x_n_1 = x_N

        # Sample x_N-1, ..., x_1 by applying posterior
        for n_int in range(self.steps - 1, 0, -1):
            n = torch.full(
                (n_samples,), n_int, device=space_bound.device, dtype=torch.long
            )
            x_n_1 = self.sample_posterior(x_n=x_n_1, n=n)

        # Sample x_0
        n = torch.full(
            (n_samples,), n_int - 1, device=space_bound.device, dtype=torch.long
        )
        x_0, _, _, _ = self.sample_x_0(n=n, x_n=x_n_1)

        return x_0

    def sample_x_0(
        self, n: TensorType[int], x_n: Batch
    ) -> Tuple[Batch, Batch, Batch, Batch]:
        """
        Sample x_0 from x_n by classifying the intersection of x_0 and x_n and sampling from the intensity.

        Parameters
        ----------
        n : TensorType[int]
            Diffusion time steps
        x_n : Batch
            Batch of data

        Returns
        -------
        Tuple[Batch, Batch, Batch, Batch]
            x_0, classified_x_0, sampled_x_0, classified_not_x_0
        """
        (
            dif_time_emb,
            event_emb,
        ) = self.compute_emb(n=n, x_n=x_n)

        # Sample x_0\x_n from intensity
        sampled_x_0 = self.intensity_model.sample(
            event_emb=event_emb,
            dif_time_emb=dif_time_emb,
            n_samples=1,
            x_n=x_n,
        )

        # Classify (x_0 ∩ x_n) from x_n
        x_n_and_x_0_logits = self.classifier_model(
            dif_time_emb=dif_time_emb, event_emb=event_emb
        )
        classified_x_0, classified_not_x_0 = x_n.thin(
            alpha=x_n_and_x_0_logits.sigmoid()
        )

        return (
            classified_x_0.add_events(sampled_x_0),
            classified_x_0,
            sampled_x_0,
            classified_not_x_0,
        )

    def sample_posterior(self, x_n: Batch, n: TensorType[int]) -> Batch:
        """
        Sample x_n-1 from x_n by predicting x_0 and then sampling from the posterior.

        Parameters
        ----------
        x_n : Batch
            Batch of data
        n : TensorType
            Diffusion time steps

        Returns
        -------
        Batch
            x_n-1
        """
        # Sample
        _, classified_x_0, sampled_x_0, classified_not_x_0 = self.sample_x_0(
            n=n, x_n=x_n
        )
        x_0_kept, _ = sampled_x_0.thin(alpha=self.retain_x0_kept[n])
        x_n_kept, _ = classified_not_x_0.thin(alpha=self.add_keep[n])

        # Put together
        x_n_1 = classified_x_0.add_events(x_n_kept).add_events(x_0_kept)

        return x_n_1

    def conditional_generation(
        self,
        condition: Batch,
        x_min: torch.Tensor,
        x_max: torch.Tensor,
    ) -> Tuple[Batch, Batch]:
        """
        Conditional sampling based on a given condition.

        Parameters
        ----------
        condition : Batch
            Condition to sample from
        x_min : torch.Tensor
            Minimum space value of the condition
        x_max : torch.Tensor
            Maximum space value of the condition

        Returns
        -------
        Tuple[Batch, Batch]
            Sampled x_0 and forecasted points
        """
        # Start with all noise (equivalent to noising our condition)
        x_next = self._sample_noise(
            space_bound=condition.space_bound,
            n_sequences=len(condition),
        )
        # Sample x_N-1, ..., x_1 by applying posterior
        for n_int in range(self.steps - 1, 0, -1):
            n = torch.full(
                (len(condition),),
                n_int,
                device=condition.space_bound.device,
                dtype=torch.long,
            )
            x_next = self.sample_posterior(x_n=x_next, n=n)

            # Only keep in forecasting zone
            _, x_next_forecast, _, _ = x_next.split_space(x_min, x_max)

            # Get noised sequence on conditional zone
            noised_train_batch, _ = self.noise(condition, n - 1)
            x_next_condition, _, _, _ = noised_train_batch.split_space(
                x_min, x_max
            )

            # Join
            x_next = x_next_forecast.add_events(x_next_condition)

        # Sample x_0
        n = torch.full(
            (len(condition),),
            int(0),
            device=condition.space_bound.device,
            dtype=torch.long,
        )
        sampled_x_0, _, _, _ = self.sample_x_0(n=n, x_n=x_next)

        # Get the forecasted points
        _, sampled_x_0_forecast, _, _ = sampled_x_0.split_space(x_min, x_max)

        # Join
        sampled_x_0 = sampled_x_0_forecast.add_events(condition)

        return sampled_x_0, sampled_x_0_forecast

    def conditional_generation_f(
        self,
        condition: Batch,
        func,
        *args,
    ) -> Tuple[Batch, Batch]:
        """
        Conditional sampling based on a conditioning function.

        Parameters
        ----------
        condition : Batch
            Condition to sample from
        func : callable
            Function to apply the condition
        *args : tuple
            Additional arguments for the function

        Returns
        -------
        Tuple[Batch, Batch]
            Sampled x_0 and forecasted points
        """
        # Start with all noise (same as noising our condition)
        x_next = self._sample_noise(
            space_bound=condition.space_bound,
            n_sequences=len(condition),
        )
        # Sample x_N-1, ..., x_1 by applying posterior
        for n_int in range(self.steps - 1, 0, -1):
            n = torch.full(
                (len(condition),),
                n_int,
                device=condition.space_bound.device,
                dtype=torch.long,
            )
            x_next = self.sample_posterior(x_n=x_next, n=n)

            # Only keep in forecasting zone
            _, x_next_forecast = x_next.split_space_func(func, *args)

            # Get noised sequence on conditional zone
            noised_train_batch, _ = self.noise(condition, n - 1)
            x_next_condition, _ = noised_train_batch.split_space_func(
                func, *args
            )

            # Join
            x_next = x_next_forecast.add_events(x_next_condition)

        # Sample x_0
        n = torch.full(
            (len(condition),),
            int(0),
            device=condition.space_bound.device,
            dtype=torch.long,
        )
        sampled_x_0, _, _, _ = self.sample_x_0(n=n, x_n=x_next)

        # Get the forecasted points
        _, sampled_x_0_forecast = sampled_x_0.split_space_func(func, *args)

        # Join
        sampled_x_0 = sampled_x_0_forecast.add_events(condition)

        return sampled_x_0, sampled_x_0_forecast
