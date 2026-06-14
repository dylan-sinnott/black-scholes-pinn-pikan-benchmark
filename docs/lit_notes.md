# Physics Informed Deep Learning: Data-driven Solutions of Nonlinear Partial Differential Equations

## Core idea
By including our prior knowledge of the system when trying to fit a ML model, for example through inclusion of a governing empirical law, the algorithim can more accurately steer itself towards a correct solution. Impact of outliers and noise in the data can be minimised. This is preferable to traditional NNs which take no consideration of the governing "physics". Using little initial and boundary data PINNs can accurately capture even non-linear behaviour. 

## Loss function
Instead of in a traditional NN where you include a term for data loss and BC/IC loss, in PINNs you also include a term for PDE residual loss, with each term assigned a weight $\lambda_i$.

## Automatic differentiation

## Strengths of PINNs
Encoding the structure of underlying empirical law through the collocation points allows for a more accurate and data-efficient learning algorithm.

No mesh is required unlike in FDMs.

## Weaknesses of PINNs
Need to use a large number of collocation points to enforce physics informed constraints on the entire spacial-temporal domain, which can be a bottleneck in high dimensional problems for which the number of collocation points required increases exponentially. 

## Useful ideas for this project
Provides a general framework for understanding PINNs, why they came about and their strengths and weaknesses.

## Open questions

# From PINNs to PIKANs: Recent Advances in Physics Informed Machine Learning (Toscano et al.)

## Core idea
PINN developed for scenarios with some (but incomplete) knowledge of governing laws and some (incomplete) measurements of state variables. Are better at dealing with noisy data and outliers than traditional NNs. Instead of in MLP frameworks where you train the weights in KANs you train the activation functions (approximated using splines).

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

