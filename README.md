# Benchmarking PINNs, PIKANs and Finite Difference Methods for Option Pricing

## Overview

This project compares classical numerical methods with physics-informed machine learning methods for solving option pricing partial differential equations.

The main case study is a European call option under the Black-Scholes model. Since the Black-Scholes equation has an analytical solution for European options, it provides a controlled benchmark for comparing numerical and neural network methods.

The methods implemented are:

- Analytical Black-Scholes formula
- Crank-Nicolson finite difference method
- Physics-Informed Neural Network (PINN)
- Physics-Informed Kolmogorov-Arnold Network (PIKAN)

The methods are compared in terms of accuracy, error behaviour, runtime and sensitivity to changes in volatility.

## Research Question

Can physics-informed neural network architectures provide competitive accuracy or efficiency compared with traditional finite difference methods for option pricing PDEs?

More specifically, this project investigates the accuracy-runtime trade-off between Crank-Nicolson, PINNs and PIKANs when applied to the Black-Scholes PDE.

## Mathematical Model

The Black-Scholes PDE for a European call option is written in time-to-maturity form as

```math
\frac{\partial V}{\partial \tau}
=
\frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
+ rS \frac{\partial V}{\partial S}
- rV.
```

The payoff condition is

```math
V(S,0)=\max(S-K,0).
```

The analytical Black-Scholes formula is used as the benchmark solution.

## Methods

### Analytical Black-Scholes Solution

The closed-form Black-Scholes call option price is used as the reference solution for evaluating the numerical and neural network approximations.

### Crank-Nicolson Method

A Crank-Nicolson finite difference solver is implemented as the classical numerical baseline. Its accuracy is tested against the analytical solution using error metrics and convergence analysis.

### Physics-Informed Neural Network

A PINN is trained to approximate the option price by minimizing a loss function containing the Black-Scholes PDE residual, payoff condition and boundary conditions.

### Physics-Informed Kolmogorov-Arnold Network

A PIKAN is implemented using the same physics-informed loss structure as the PINN, but with a Kolmogorov-Arnold Network architecture. Its performance is compared against both the baseline PINN and the Crank-Nicolson method.

## Evaluation Metrics

The following metrics are used to compare model predictions against the analytical Black-Scholes solution:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Maximum Absolute Error
- Relative L2 Error
- Runtime / training time

## Key Findings

The Crank-Nicolson method achieved the best accuracy-runtime trade-off for the one-dimensional Black-Scholes problem. It produced very low errors with substantially shorter runtime than the neural network methods.

The baseline PINN was able to learn the general shape of the Black-Scholes price surface, but its errors were larger than those of the Crank-Nicolson method.

The PIKAN improved substantially on the baseline PINN in terms of accuracy, but required longer training time. It did not outperform Crank-Nicolson in this controlled one-dimensional setting.

Overall, the results suggest that physics-informed neural networks are flexible PDE solvers, but for this classical Black-Scholes problem, traditional finite difference methods remain more efficient.

## Sensitivity Analysis

A volatility sensitivity analysis is performed by varying the volatility parameter while keeping the other Black-Scholes parameters fixed.

This tests whether the relative performance of Crank-Nicolson, PINN and PIKAN methods remains stable across different volatility regimes.

## Conference talk

Presented at **PyCon Ireland 2026**, Dublin,
21 November 2026.

**Talk:** “Can Machine-Learning Models beat Classical Option Pricing?”

The talk presented this project's comparison of Crank–Nicolson,
PINN and PIKAN methods for Black–Scholes option pricing.

## Repository Structure

```text
docs/
    Research notes, experiment logs and methodology notes

notebooks/
    Analytical, Crank-Nicolson, PINN, PIKAN, comparison and sensitivity analysis notebooks

report/
    Final report, figures and LaTeX source

results/
    Saved figures and tables

src/
    Core implementations for analytical pricing, finite difference solvers,
    neural network models, loss functions, metrics and evaluation utilities

