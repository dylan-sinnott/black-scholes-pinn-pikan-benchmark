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

# A comprehensive and FAIR comparison between MLP and KAN representation for differential equations and operator networks

## Key findings
KANs based on B-splines lack accuracy and efficiency but modified versions based on low order orthogonal polynomials have comparable performance to PINNS.

This study compares PINNs and PIKANs using state-of-the-art optimisation techniques (residual-based attention and eddy viscosity formulations) enabling the models to solve more complex problems.

**Residual based attention:** Addresses the issue of the residuals potentially getting overlooked when calculating the cumulative loss function. Scaling the loss terms using local multipliers (particularly residual-based attention weights, which are based on the exponentially weighted moving average of the residuals) balances the contribution of individual training points within each loss term introducing "residual homogeneity." The update rule for RBA for any training point i on iteration k is: 

$$
\alpha_i^{k+1}
=
(1-\eta^*)\alpha_i^k
+
\eta^*
\frac{|e_i|}{\|e\|_{\infty}}
$$

where $e_i$ is the residual at point $i$, $\eta^*$ is the learning rate, and $\|e\|_{\infty}$ is the maximum absolute residual.

## Advantages of KANs
Showed promise in rectifying MLP issues such as interpretability and catastrophic forgetting in supervised and unsupervised tasks.

cPIKANs are the combination of physics informed machine learning and the Chebyshev KAN. These don't require grid points which reduces the number of trainable parameters.

Liu et al. showed KANs to significantly outperform MLP in solving the 2D Poisson equation (albeit with shallow networks).

On a discontinuous and oscillatory function, KANs outperformed MLP in terms of L2 error but took far longer to implement. Modified cKANs achieved comparable L2 error though the MLP was slightly more efficient in runtime. 

On the Helmholtz equation Vanilla PIKAN outperforms vanilla PINN but with drastically higher iteration time. For a higher wave number, cPIKAN + RBA performed better than PINN with fewer parameters and faster iteration time (ADAM optimiser). With an unbiased loss function, ADAM optimiser (and lower wave number) cPIKAN + RBA also performed best with far better performance than LBFGS optimiser and a biased loss function, although with more iterations.

On NS (Navier-Stokes equation) with Reynolds number (Re) 400, cPIKAN is the best choice (with RBA).

## Disadvantages of KANs
Using learnable B-splines as activation functions significantly increases computational cost.

Unmodified (to fix this compose each cKAN layer with a tanh function, except for the last layer) cKAN training became unstable on the discontinuous, oscillatory function after 2000 iterations. 

cPIKANs perform better than PINNs and reduce the computational overhead of PIKANs but induces more oscillations and can become unstable.

For NS equation with high Re PINNs dramatically outperform cPIKANs, though by introducing EVM, cPIKANs + EVM outperform PINNs and PINNs + EVM (or RBA).

Cost of training PIKANs based on Jacobi polynomials can be several times more than that of PINNs.

On the 2D non-linear Allen Cahn equation PINN + RBA outperforms all PIKAN/cPIKAN architecture in terms of L2 and training time, albeit with significantly more parameters than processes that don't include RBA.

## Results relevant to PDEs
Instead of using learnable B-splines use radial basis function, wavelets or Jacobi polynomials (e.g Chebyshev, Legendre). 

ADAM optimiser works best on most examples listed.

## Implications for my project
cPIKANs perform better than PIKANs but can be unstable when not combined with RBA. RBA/EVM should be included in all examples since it consistently improved performance significantly, albeit with the trade off of introducing many more hyperparameters.

## Open questions
Will the Black-Scholes equation exhibit behavior similar to previous equations?
How will training time/hyperparameters contrast between PINNs and PIKANs on BS?

# Physics Informed Neural Network for Option Pricing 

## Black-Scholes formulation
I will consider only EU options in my project (for now), so consider the sections of this paper with the EU BS model, namely that of European Call options (and American Call, since excercising these early is "never considered reasonable").

## PINN implementation
This study made use of vanilla PINNs.

## Training setup
This study favours wider networks over deeper ones, integrates residual connections to facilitate gradient flow and adopts the tanh() activation function. 25,000 epochs of training were used.

## Evaluation metrics
To validate the accuracy of EU Calls they are compared against the analytical solution. Total Loss, correlation between $y_{true}$ and $y_{PINN}$ and MSE are computed.

## Results
On simulated data the initial 10,000 epochs converge effectively but there are potential convergence challenges after this, particularly for spot prices approaching 0 or the boundary value 1. However, the PINN model does adeptly capture the shape of the EU Call.

On market data (EU Call) PINN performs better than the analytical solution but still have errors compared to market price. But there are high correlations between market and predicted price.

## Limitations
Convergence challenges do exist for PINNs, necessitating careful consideration of various parameters such as volatility/risk free rate. The selection of these significantly impacts model performances.

## Ideas for my project
Use comparision against analytical solution, MSE, total loss and correlation between true/PINN predicted solutions as evaluation metrics. 

# Numerical Pricing of European Stock Option Based on Black-Scholes Model Using Crank-Nicolson Method

## PDE
Use the BS PDE, derived from the assumption that the underlying stock price evolves according to GBM. We use the transformation $\tau = T - t$.

## Terminal condition
$$V(S,T) = max(S-K,0)$$

## Boundary conditions
$$V(0,t) = 0, V(S_{max},t) = S_{max} - K e^{-r(T-t)}$$

Here $S_{max}=4K$, large enough to ensure upper bound solution approaches theoretical asymptotic value.

## Computational domain
Uniform spatial and temporal grid. $[0,S_{max}]$ is discretised into M uniformly spaced grid points with step size $\Delta S = \frac{S_{max}}{M}$. $[0, T]$ is split into N uniform time steps with step size $\Delta \tau = \frac{T}{N}$.

## Discretization
Second order accurate central FDM schemes are employed for spatial derivatives. Crank-Nicolson is used to estimate temporal derivatives.

## Linear system
Tridiagonal system solved at each timestep using Thomas' Algorithm, as it is computationally simple with good stability for tridiagonal systems.

## Validation
Compared against analytical Black-Scholes price. Sensitivity analysis to volatility and temporal covergence tests to demonstrate consistency of numerical solution's behaviour with variation of key parameters.

## Ideas for my project
Will serve as classical benchmark against PINN and PIKAN.
