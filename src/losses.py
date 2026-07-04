# losses.py

import torch


def black_scholes_pde_loss(
    model,
    training_data,
    r,
    sigma,
    K,
    weight_pde=1.0,
    weight_ic=1.0,
    weight_lb=1.0,
    weight_ub=1.0
):
    (
        S_f, tau_f,
        S_ic, tau_ic, V_ic,
        S_lb, tau_lb, V_lb,
        S_ub, tau_ub, V_ub
    ) = training_data

    residual = model.pde_residual(S_f, tau_f, r, sigma)
    loss_pde = torch.mean((residual / K) ** 2)

    V_ic_pred = model(S_ic, tau_ic)
    loss_ic = torch.mean(((V_ic_pred - V_ic) / K) ** 2)

    V_lb_pred = model(S_lb, tau_lb)
    loss_lb = torch.mean(((V_lb_pred - V_lb) / K) ** 2)

    V_ub_pred = model(S_ub, tau_ub)
    loss_ub = torch.mean(((V_ub_pred - V_ub) / K) ** 2)

    total_loss = (
        weight_pde * loss_pde
        + weight_ic * loss_ic
        + weight_lb * loss_lb
        + weight_ub * loss_ub
    )

    return total_loss, loss_pde, loss_ic, loss_lb, loss_ub