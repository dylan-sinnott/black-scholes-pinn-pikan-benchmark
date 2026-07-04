import torch
import torch.nn as nn

from kan import KAN


class BlackScholesPIKAN(nn.Module):
    """
    Physics-informed KAN wrapper for the Black-Scholes PDE.

    This class is designed to behave like your BlackScholesPINN class:
    - forward(S, tau) returns the model option price V(S, tau)
    - pde_residual(S, tau, r, sigma) returns the Black-Scholes PDE residual

    Internally, it uses PyKAN's spline-based KAN model.
    """

    def __init__(
        self,
        layers,
        grid=5,
        k=None,
        degree=3,
        S_max=200.0,
        T=1.0,
        seed=0,
        device="cpu",
    ):
        super().__init__()

        if k is None:
            k = degree

        self.layers = layers
        self.grid = grid
        self.k = k
        self.S_max = S_max
        self.T = T
        self.seed = seed
        self.device = device

        self.network = KAN(
            width=layers,
            grid=grid,
            k=k,
            seed=seed,
            device=device,
            grid_range=[0.0, 1.0],
            symbolic_enabled=False,
            save_act=False,
            auto_save=False,
        )

    def forward(self, S, tau):
        S_scaled = S / self.S_max
        tau_scaled = tau / self.T

        x = torch.cat((S_scaled, tau_scaled), dim=1)

        return self.network(x)

    def pde_residual(self, S, tau, r, sigma):
        S = S.clone().detach().requires_grad_(True)
        tau = tau.clone().detach().requires_grad_(True)

        V = self.forward(S, tau)

        V_tau = torch.autograd.grad(
            V,
            tau,
            grad_outputs=torch.ones_like(V),
            create_graph=True,
        )[0]

        V_S = torch.autograd.grad(
            V,
            S,
            grad_outputs=torch.ones_like(V),
            create_graph=True,
        )[0]

        V_SS = torch.autograd.grad(
            V_S,
            S,
            grad_outputs=torch.ones_like(V_S),
            create_graph=True,
        )[0]

        residual = (
            V_tau
            - 0.5 * sigma**2 * S**2 * V_SS
            - r * S * V_S
            + r * V
        )

        return residual