import torch
import torch.nn.functional as F


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """Compute betas for a given alpha_t_bar function.

    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    We use it to implement the cosine beta schedule from https://arxiv.org/abs/2112.10741

    Parameters
    ----------
    num_diffusion_timesteps : int
        Number of diffusion timesteps
    alpha_bar : callable
        Function that returns the cumulative product of (1-beta) over time from
        t = [0,1]
    max_beta : float
        Maximum value for beta

    Returns
    -------
    betas : torch.Tensor
        Beta values for each timestep
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)


def thin_and_add_probs(alpha_bar):
    retain = F.pad(alpha_bar, (0, 1), value=0)  # alpha bar
    add_prob = 1 - retain  # 1 - alpha_bar

    # We introduce the pre versions to index throughout the backward process using n and not n-1
    add_pre = add_prob[:-1]
    add_prob = add_prob[1:]

    retain_pre = retain[:-1]
    retain = retain[1:]

    # Posterior probability of keeping the atom until x_t-1
    retain_x0_kept = (retain_pre - retain) / (1 - retain)

    # Posterior probability of keeping the added atoms
    add_keep = add_pre / add_prob

    return (
        retain.float(),
        add_prob.float(),
        add_keep.float(),
        retain_x0_kept.float(),
    )
