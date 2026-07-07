# 13/06/2026

## Progress
- Created Github repository.
- Created project structure.
- Read through "From PINNs to PIKANs."
- Skimmed "Physics Informed Deep Learning" (Raissi et al.)

## Next steps
- Read "Physics Informed Deep Learning" (Raissi et al.)
- Read up on FDMs, specifically Crank-Nicholson.
- Read literature applying PIML techniques to the Black-Scholes equation.

# 14/06/2026

## Progress
- Read "Physics Informed Deep Learning"

## Next steps
- Read up on FDMs, specifically Crank-Nicholson.
- Read literature applying PIML techniques to the Black-Scholes equation.
- Create README

# 17/06/2026

## Progress
- Read "Physics Informed Machine Learning for Option Pricing"
- Read "A comprehensive and FAIR comparision between MLP and KAN representations for differential equations and operator networks"
- Read "Numerical Pricing of European Stock Options based on Black-Scholes Model using Crank-Nicolson method"
- Completed (initial) project README

## Next steps
- Implement analytical Black-Scholes solution.
- Implement Crank-Nicolson solver.

# 21/06/2026

## Progress
- Implemented call_price and put_price functions.
- Tested these in a notebook by verifying put-call parity and generating a price surface.

## Next steps.
- Implement Crank-Nicolson solver.
- Test the Crank-Nicolson solver: generate convergence plots, error plots, verify boundary conditions etc.

# 30/06/2026

## Progress
- Implemented Crank-Nicolson solver. 

## Next steps
- Test the Crank-Nicolson solver: generate convergence plots, error plots, verify boundary conditions etc.

# 03/07/2026

## Progress
- Tested the Crank-Nicolson solver.
- Created metrics.py and plotting.py.
- Created pinn.py and implemented a basic PINN.
- Created pinn.ipynb, trained the PINN on one parameter set and created a loss plot. 

## Next steps
- Generate PINN predictions over a grid of stock prices and times to maturity. 
- Compare PINN output to analytical Black-Scholes prices. Generate PINN price and error plots and a metrics table with RMSE, MAE, maximum absolute error, relative L2 error.
- Tune PINN training. 


# 04/07/2026

## Progress
- Tuned PINN output.
- Created PIKAN.
- Did a basic comparison between Crank-Nicolson, PINNs and PIKANs.

## Next steps
- Run some market parameter sensitivity testing.

# 05/07/2026

## Progress
- Tuned PIKAN
- Ran sensitivity analysis

# 06/07/2026

## Progress
- Finalised figures and tables
- Wrote report
- Polished README
