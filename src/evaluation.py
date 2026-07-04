# evaluation.py

import numpy as np
import torch

from src.analytical import black_scholes_call_grid
from src.metrics import mse, rmse, mae, max_abs_error, relative_l2_error


def evaluate_model_on_grid(model, S_grid, tau_grid, K, r, sigma):
    model.eval()

    if torch.is_tensor(S_grid):
        S_np = S_grid.detach().cpu().numpy()
    else:
        S_np = np.asarray(S_grid, dtype=float)

    if torch.is_tensor(tau_grid):
        tau_np = tau_grid.detach().cpu().numpy()
    else:
        tau_np = np.asarray(tau_grid, dtype=float)

    device = next(model.parameters()).device

    S_flat = torch.tensor(
        S_np.reshape(-1, 1),
        dtype=torch.float32,
        device=device
    )

    tau_flat = torch.tensor(
        tau_np.reshape(-1, 1),
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        pred = model(S_flat, tau_flat)
        pred = pred.detach().cpu().numpy().reshape(S_np.shape)

    true = black_scholes_call_grid(S_np, tau_np, K, r, sigma)

    return pred, true


def compute_metric_row(
    model_name,
    model,
    S_grid,
    tau_grid,
    K,
    r,
    sigma,
    architecture="",
    scheduler="",
    loss_type=""
):
    pred, true = evaluate_model_on_grid(
        model=model,
        S_grid=S_grid,
        tau_grid=tau_grid,
        K=K,
        r=r,
        sigma=sigma
    )

    return {
        "Model": model_name,
        "Architecture": architecture,
        "Scheduler": scheduler,
        "Loss": loss_type,
        "MSE": mse(pred, true),
        "RMSE": rmse(pred, true),
        "MAE": mae(pred, true),
        "Max Abs Error": max_abs_error(pred, true),
        "Relative L2 Error": relative_l2_error(pred, true)
    }