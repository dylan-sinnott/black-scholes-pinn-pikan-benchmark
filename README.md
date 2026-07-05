# Benchmarking PINNs, PIKANs and Finite Difference Methods for Option Pricing

## Overview

This project compares classical numerical methods and scientific machine learning methods for solving option pricing partial differential equations.

The main case study is the European call option under the Black-Scholes model, where an analytical solution is available. This allows the numerical and neural network methods to be benchmarked against a known exact solution.

The methods implemented are:

- Analytical Black-Scholes formula
- Crank-Nicolson finite difference method
- Physics-Informed Neural Network (PINN)
- Physics-Informed Kolmogorov-Arnold Network (PIKAN)

The project compares the methods in terms of accuracy, error behaviour, runtime and robustness under changes in volatility.

## Research Question

Can physics-informed neural network architectures provide competitive accuracy or efficiency compared with traditional finite difference methods for option pricing PDEs?

More specifically, this project investigates the accuracy-runtime tradeoff between Crank-Nicolson, PINNs and PIKANs when applied to the Black-Scholes PDE.

## Mathematical Model

The Black-Scholes PDE for a European call option is written in time-to-maturity form as

$$
\frac{\partial V}{\partial \tau}
=
\frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
+
rS \frac{\partial V}{\partial S}
-
rV.
$$

The terminal payoff condition is

$$
V(S,0) = \max(S-K,0).
$$

The analytical Black-Scholes formula is used as the benchmark solution.

## Methods

### Analytical Black-Scholes Solution

The closed-form Black-Scholes call option price is used as the ground truth for evaluating numerical and neural network approximations.

### Crank-Nicolson Method

A Crank-Nicolson finite difference solver is implemented as the classical numerical baseline. Its accuracy is tested against the analytical solution using error metrics and convergence analysis.

### Physics-Informed Neural Network

A PINN is trained to approximate the option price by minimizing a loss function containing the Black-Scholes PDE residual, payoff condition and boundary conditions.

### Physics-Informed Kolmogorov-Arnold Network

A PIKAN architecture is implemented using the same Black-Scholes PDE residual and boundary/initial condition structure as the PINN. Its performance is compared against the PINN and Crank-Nicolson methods.

## Evaluation Metrics

The following metrics are used to compare model predictions against the analytical Black-Scholes solution:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Maximum Absolute Error
- Relative L2 Error
- Runtime / training time

## Sensitivity Analysis

A volatility sensitivity analysis is performed by varying the volatility parameter while keeping the other Black-Scholes parameters fixed.

This is used to test whether the relative performance of Crank-Nicolson, PINN and PIKAN methods remains stable across different volatility regimes.

## Repository Structure

```text
docs/
    Research notes, experiment logs and methodology notes

notebooks/
    Analytical, Crank-Nicolson, PINN, PIKAN, comparison and sensitivity analysis notebooks

src/
    Core implementations for analytical pricing, finite difference solvers,
    neural network models, loss functions, metrics and evaluation utilities    

report/
    Final report, figures and written analysis

results/
    Figures, tables.
