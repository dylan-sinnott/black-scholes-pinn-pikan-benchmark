"""In this file, I will implement the Crank-Nicolson method for solving the Black-Scholes PDE."""

import numpy as np

import numpy as np


class Crank_Nicolson_Solver:

    def __init__(self, S0, K, T, r, sigma, S_max, M, N):
        # Store parameters

        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.S_max = S_max
        self.M = M
        self.N = N


    def _setup_grid(self):
        self.S = np.linspace(0, self.S_max, self.M + 1)
        self.tau = np.linspace(0, self.T, self.N + 1)

        self.dS = self.S_max / self.M
        self.dt = self.T / self.N


    def _setup_coefficients(self):
        i = np.arange(1, self.M)

        self.alpha = 0.25 * self.dt * (self.sigma**2 * i**2 - self.r * i)
        self.beta = -0.5 * self.dt * (self.sigma**2 * i**2 + self.r)
        self.gamma = 0.25 * self.dt * (self.sigma**2 * i**2 + self.r * i)


    def _setup_matrices(self):

        # Here I build matrices A and B.

        size = self.M - 1

        self.A = np.zeros((size, size))
        self.B = np.zeros((size, size))

        for j in range(np.size(self.beta)):
            self.A[j, j] = 1 - self.beta[j]
            self.B[j, j] = 1 + self.beta[j]

            if j > 0:
                self.A[j, j - 1] = -self.alpha[j]
                self.B[j, j - 1] = self.alpha[j]

            if j < size - 1:
                self.A[j, j + 1] = -self.gamma[j]
                self.B[j, j + 1] = self.gamma[j]


    def _initial_condition(self):
        return np.maximum(self.S - self.K, 0)


    def _left_boundary(self, tau_value):
        return 0


    def _right_boundary(self, tau_value):
        return self.S_max - self.K * np.exp(-self.r * tau_value)


    def solve(self):
        self._setup_grid()
        self._setup_coefficients()
        self._setup_matrices()

        V = self._initial_condition()

        for n in range(self.N):
            tau_old = self.tau[n]
            tau_new = self.tau[n + 1]

            V_left_old = self._left_boundary(tau_old)
            V_left_new = self._left_boundary(tau_new)

            V_right_old = self._right_boundary(tau_old)
            V_right_new = self._right_boundary(tau_new)

            rhs = self.B @ V[1:self.M]

            rhs[0] += self.alpha[0] * (V_left_old + V_left_new)
            rhs[-1] += self.gamma[-1] * (V_right_old + V_right_new)

            V[1:self.M] = np.linalg.solve(self.A, rhs)

            V[0] = V_left_new
            V[-1] = V_right_new

        self.V = V
        self.price = self.interpolate_price()

        return self.price, self.S, self.V


    def interpolate_price(self):
        return np.interp(self.S0, self.S, self.V)
