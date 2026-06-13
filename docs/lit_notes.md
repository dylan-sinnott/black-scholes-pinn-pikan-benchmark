# Physics Informed Deep Learning: Data-driven Solutions of Nonlinear Partial Differential Equations

## Core idea

## Loss function

## Automatic differentiation

## Strengths of PINNs

## Weaknesses of PINNs

## Useful ideas for this project

## Open questions

# From PINNs to PIKANs: Recent Advances in Physics Informed Machine Learning (Toscano et al.)

## Core idea
PINN developed for scenarios with some (but incomplete) knowledge of governing laws and some (incomplete) measurements of state variables. Are better at dealing with noisy data and outliers than traditional NNs.

## PINN limitations
Spectral bias (weaknesses in PIML models in learning high frequency functions). 

Can struggle with sharp gradients.

Residuals at key points can be underrepresented in the overall sum of objective function, so that even though the total loss might decrease during training, certain spacial/temporal characteristics might not be fully captured.

## PIKAN advantages
More robust to noise and can lead to improved performance over PINNs with fewer parameters. You can add residual connections which enable accurate calculation of higher order derivatives to improve the PIML architecture performance. 

Have the potential to converge faster.

PIKANs can have lower computational cost than PINNs while maintaining good accuracy.

## Useful ideas for this project
Motivating the use of PINNs.

Discussion section.

Lit review.

## Open questions
On Black-Scholes specifically, do PIKANs outperform PINNs?

