from collections import deque
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn


def update_params(
    optim: torch.optim.Optimizer,
    loss: Tensor,
    retain_graph: bool=False,
) -> None:
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)  # type: ignore[no-untyped-call]
    optim.step()


def disable_gradients(network: nn.Module) -> None:
    # Disable calculations of gradients.
    for param in network.parameters():
        param.requires_grad = False


def calculate_huber_loss(td_errors: Tensor, kappa: float=1.0) -> Tensor:
    """
    Calculate huber loss element-wisely depending on kappa.
    if |td_errors| ≤ kappa:
        huber_loss = 0.5 * td_errors ** 2
    else:
        huber_loss = kappa * (|td_errors| - 0.5 * kappa)
    """
    loss = torch.where(
        td_errors.abs() <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (td_errors.abs() - 0.5 * kappa)
    )
    return loss


def calculate_quantile_huber_loss(td_errors: Tensor, taus: Tensor, kappa: float=1.0) -> Tensor:
    assert not taus.requires_grad
    batch_size, N, N_dash = td_errors.shape

    # Calculate huber loss element-wisely.
    element_wise_huber_loss = calculate_huber_loss(td_errors, kappa)
    assert element_wise_huber_loss.shape == (batch_size, N, N_dash)

    # Calculate quantile huber loss element-wisely.
    # see: https://arxiv.org/pdf/1710.10044.pdf -> equation (10)
    element_wise_quantile_huber_loss = torch.abs(
        taus[..., None] - (td_errors.detach() < 0).float()
    ) * element_wise_huber_loss / kappa
    assert element_wise_quantile_huber_loss.shape == (batch_size, N, N_dash)

    # Quantile huber loss.
    batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(dim=1).mean(dim=1, keepdim=True)
    assert batch_quantile_huber_loss.shape == (batch_size, 1)
    
    quantile_huber_loss = batch_quantile_huber_loss.mean()

    return quantile_huber_loss


def evaluate_quantile_at_action(s_quantiles: Tensor, actions: Tensor) -> Tensor:
    assert s_quantiles.shape[0] == actions.shape[0]

    batch_size = s_quantiles.shape[0]
    N = s_quantiles.shape[1]

    # Expand actions into (batch_size, N, 1).
    action_index = actions.unsqueeze(-1).unsqueeze(-1).expand(batch_size, N, 1)

    # Calculate quantile values at specified actions.
    sa_quantiles = s_quantiles.gather(dim=2, index=action_index)

    return sa_quantiles
