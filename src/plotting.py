# This file contains reusable plotting functions for the project.

import matplotlib.pyplot as plt
import numpy as np


def plot_price_comparison(
    S,
    numerical,
    analytical,
    title="Numerical vs Analytical Option Price",
    xlabel="Stock Price S",
    ylabel="Option Value V",
):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(S, numerical, label="Numerical", linewidth=2)
    ax.plot(S, analytical, "--", label="Analytical", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig, ax


def plot_error(
    S,
    error,
    title="Error Between Numerical and Analytical Prices",
    xlabel="Stock Price S",
    ylabel="Error",
):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(S, error, label="Error", color="red", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig, ax


def plot_price_surface(
    S_grid,
    t_grid,
    V,
    title="Option Price Surface",
    xlabel="Stock Price S",
    ylabel="Time t",
    colorbar_label="Option Value V",
    levels=50,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    contour = ax.contourf(S_grid, t_grid, V, levels=levels, cmap="viridis")
    fig.colorbar(contour, ax=ax, label=colorbar_label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    return fig, ax


def plot_absolute_error_surface(
    S_grid,
    t_grid,
    absolute_error,
    title="Absolute Error Surface",
    xlabel="Stock Price S",
    ylabel="Time t",
    colorbar_label="Absolute Error",
    levels=50,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    contour = ax.contourf(S_grid, t_grid, absolute_error, levels=levels, cmap="magma")
    fig.colorbar(contour, ax=ax, label=colorbar_label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    return fig, ax


def plot_relative_error_surface(
    S_grid,
    t_grid,
    relative_error,
    title="Relative Error Surface",
    xlabel="Stock Price S",
    ylabel="Time t",
    colorbar_label="Relative Error",
    levels=50,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    contour = ax.contourf(S_grid, t_grid, relative_error, levels=levels, cmap="plasma")
    fig.colorbar(contour, ax=ax, label=colorbar_label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    return fig, ax


def plot_convergence(
    grid_values,
    errors,
    title="Convergence Plot",
    xlabel="Grid Spacing",
    ylabel="Error",
    reference_order=None,
    grid_values_are_spacings=True,
):
    grid_values = np.asarray(grid_values)
    errors = np.asarray(errors)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(grid_values, errors, "o-", label="Observed error", linewidth=2)

    if reference_order is not None:
        if grid_values_are_spacings:
            reference = errors[0] * (grid_values / grid_values[0]) ** reference_order
        else:
            reference = errors[0] * (grid_values / grid_values[0]) ** (-reference_order)

        ax.loglog(
            grid_values,
            reference,
            "--",
            label=f"Order {reference_order} reference",
            linewidth=2,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig, ax


def plot_training_loss(
    losses,
    title="Training Loss",
    xlabel="Epoch",
    ylabel="Loss",
):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(losses, linewidth=2, label="Training loss")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig, ax


def plot_loss_components(
    history,
    title="PINN Loss Components",
    xlabel="Epoch",
    ylabel="Loss",
):
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, values in history.items():
        ax.semilogy(values, label=name, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig, ax


def plot_collocation_points(
    S_points,
    t_points,
    title="PINN Collocation Points",
    xlabel="Stock Price S",
    ylabel="Time t",
):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(S_points, t_points, s=10, alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax


def plot_pde_residual_surface(
    S_grid,
    t_grid,
    residual,
    title="PDE Residual Surface",
    xlabel="Stock Price S",
    ylabel="Time t",
    colorbar_label="PDE Residual",
    levels=50,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    contour = ax.contourf(S_grid, t_grid, residual, levels=levels, cmap="coolwarm")
    fig.colorbar(contour, ax=ax, label=colorbar_label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    return fig, ax