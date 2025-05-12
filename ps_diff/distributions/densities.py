import torch
import torch.distributions as D

from ps_diff.distributions.mvnorm.multivariate_normal_cdf import (
    multivariate_normal_cdf,
)


class MultivariateNormal(D.MultivariateNormal):
    def __init__(
        self,
        mean: torch.Tensor,
        cov_diag: torch.Tensor,
    ) -> None:
        """
        Instantiate multivariate normal with specific mean and covariance matrix.

        Parameters:
        ----------
        mean : torch.Tensor
            Mean of the multivariate normal distribution.
        cov_diag : torch.Tensor
            Covariance matrix diagonal of the multivariate normal distribution.
            Here we assume independence of the dimensions and get a diagonal covariance matrix.
        """
        # TODO might want to change cov and mean parametrisation and dependence between dimensions
        super().__init__(
            loc=mean.tanh(),
            covariance_matrix=torch.diag_embed(torch.exp(-torch.abs(cov_diag)) + 1e-3),
            validate_args=False,
        )

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Change cdf to be on [-1, x]^d.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor.

        Returns:
        -------
        torch.Tensor
            Truncated CDF of the input tensor.
        """

        return multivariate_normal_cdf(
            x, self.loc, self.covariance_matrix
        ) - multivariate_normal_cdf(
            torch.ones_like(x) * (-1), self.loc, self.covariance_matrix
        )


DISTRIBUTIONS = {"multivar_normal": MultivariateNormal}
