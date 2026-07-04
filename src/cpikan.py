import torch
import torch.nn as nn

class BlackScholesCPIKAN(nn.Module): # Using a Chebyshev Kolmogorov Arnold Neural Network (PIKAN) architecture for the Black-Scholes PDE

    def __init__(self, layers, S_max, T, price_scale=100.0):
        super().__init__()

        self.S_max = S_max
        self.T = T
        self.price_scale = price_scale

        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))

            if i < len(layers) - 2:
                modules.append(nn.Tanh())

        self.network = nn.Sequential(*modules)

    def forward(self, S, tau):
        S_norm = S / self.S_max
        tau_norm = tau / self.T

        x = torch.cat((S_norm, tau_norm), dim=1)

        return self.price_scale * self.network(x)

    def pde_residual(self, S, tau, r, sigma):
        S = S.detach().clone().requires_grad_(True)
        tau = tau.detach().clone().requires_grad_(True)

        V = self.forward(S, tau)

        V_tau = torch.autograd.grad(
            V,
            tau,
            grad_outputs=torch.ones_like(V),
            create_graph=True
        )[0]

        V_S = torch.autograd.grad(
            V,
            S,
            grad_outputs=torch.ones_like(V),
            create_graph=True
        )[0]

        V_SS = torch.autograd.grad(
            V_S,
            S,
            grad_outputs=torch.ones_like(V_S),
            create_graph=True
        )[0]

        residual = V_tau - 0.5 * sigma**2 * S**2 * V_SS - r * S * V_S + r * V

        return residual
    
    def prediction(self, S, tau):
        self.eval()
        with torch.no_grad():
            return self.forward(S, tau)
     