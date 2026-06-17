# Benchmarking Physics-Informed Neural Networks and Finite Difference Methods for Option Pricing

## Overview
This project investigates the use of scientific machine learning methods for solving option pricing partial differential equations.

The primary objective is to compare classical numerical methods with modern physics-informed neural networks when pricing European options under the Black-Scholes framework.

Methods considered include:
- Analytical Black-Scholes solution.
- Finite Difference Methods, namely Crank-Nicolson.
- Physics-Informed Neural Networks (PINNs)
- Physics-Informed Kolmogorov-Arnold Networks (PIKANs)

## Research Question
Can modern physics-informed neural network architectures provide accuracy or efficiency advantages over traditional numerical methods for option pricing?

## Objectives
- Implement analytical Black-Scholes pricing.
- Implement Crank-Nicolson.
- Implement PINN-based solvers.
- Implement PIKAN architectures.
- Compare accuracy, convergence behaviour and computational cost.
- Benchmark all methods on European option pricing problems.

## Repository Structure
docs/
Research notes and methodology

notebooks/
Exploratory notebooks and experiments

src/
Core implementations

tests/
Validation and testing

report/
Final report and figures

## Current Status
- [x] Literature review 
- [ ] Analytical Black-Scholes implementation
- [ ] Crank-Nicolson implementation
- [ ] PINN implementation
- [ ] PIKAN implementation
- [ ] Benchmarking
- [ ] Final report

## Literature
Key references  include:
- Physics Informed Deep Learning: Data-driven Solutions of Nonlinear Partial Differential Equations (Raissi et al.)
- Physics Informed Neural Network for Option Pricing (Dhiman and Hu)
- From PINNs to PIKANs: Recent Advances in Physics-Informed Machine Learning (Toscano et al.)
- A comprehensive and FAIR comparision between MLP and KAN representations for differential equations and operator networks (Shukla et al.)
- Numerical Pricing of European Stock Option Based on Black-Scholes Model Using Crank-Nicolson Method (Putri and Artiono)

## Author
Dylan Sinnott

Incoming MSc Quantitative Finance Student
UCD Michael Smurfit Graduate Business School
