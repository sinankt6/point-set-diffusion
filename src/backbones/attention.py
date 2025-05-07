import torch.nn as nn
import torch

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from src.data import Batch

patch_typeguard()


@typechecked
class AttentionPointEmb(nn.Module):
    """
    Self-attention for point embedding.

    Parameters:
    ----------
    n_blocks : int
        Number of transformer blocks
    input_dim : int
        Input dimension
    embs_dim : int
        Output dimension
    """

    def __init__(self, n_blocks, input_dim, embs_dim) -> None:
        super().__init__()
        # Instantiate projection layer if necessary
        if input_dim != embs_dim:
            self.initial_projection = nn.Sequential(
                nn.Linear(input_dim, embs_dim * 2),
                nn.ReLU(),
                nn.Linear(embs_dim * 2, embs_dim),
            )
            for m in self.initial_projection.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(
                        m.weight, mode="fan_in", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # Instantiate transformer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embs_dim,
            nhead=int(1),
            dim_feedforward=4 * int(embs_dim),
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=n_blocks,
        )

        # Save dimensions
        self.input_dim = input_dim
        self.embs_dim = embs_dim

    def forward(
        self,
        x_n: Batch,
    ) -> TensorType[float, "batch", "sequence", "embs_dim"]:
        """
        Parameters:
        ----------
        x_n : Batch
            Input batch with points and mask
        Returns:
        -------
        x : TensorType[float, "batch", "sequence", "embs_dim"]
            Output sequence embedding
        """
        # get data from batch
        x = x_n.points
        mask = x_n.mask[:, :, 0].bool()

        # Initial projection with MLP
        if self.input_dim != self.embs_dim:
            x = self.initial_projection(x)

        # Mask points
        x = x * mask[..., None]

        # Encode with transformer
        x = self.encoder_layer(
            src=x,
            src_key_padding_mask=~mask,
        )

        # Empty sequences return nan values, zero embedding as a solution
        assert not (
            (mask.sum(-1) != 0) == torch.isnan(x).sum(-1).sum(-1)
        ).any(), "Nan values are not only in empty sequences"
        x = torch.nan_to_num(x, nan=0.0)

        return x
