# training_points.py

import torch


def generate_training_points(N_f, N_ic, N_bc, S_max, T, K, r):
    S_f = torch.rand(N_f, 1) * S_max
    tau_f = torch.rand(N_f, 1) * T

    S_ic = torch.rand(N_ic, 1) * S_max
    tau_ic = torch.zeros_like(S_ic)
    V_ic = torch.clamp(S_ic - K, min=0.0)

    tau_lb = torch.rand(N_bc, 1) * T
    S_lb = torch.zeros_like(tau_lb)
    V_lb = torch.zeros_like(tau_lb)

    tau_ub = torch.rand(N_bc, 1) * T
    S_ub = torch.full_like(tau_ub, S_max)
    V_ub = S_max - K * torch.exp(-r * tau_ub)

    return (
        S_f, tau_f,
        S_ic, tau_ic, V_ic,
        S_lb, tau_lb, V_lb,
        S_ub, tau_ub, V_ub
    )