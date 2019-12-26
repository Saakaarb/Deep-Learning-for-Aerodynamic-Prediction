# Deep-Learning-for-Aerodynamic-Prediction
This repository contains code used to create and train a deep neural network that replicates a RANS solver for aerodynamic prediction over airfoils.

## Files:

- Front_end.py: All Major training parameters are modified in this file
- back_end.py: Major functions including those to build and train network based on specified hyper-parameters are included in this function
- Data Files(Y):
  - Output_rho.mat: Grid-density values for each of the training/validation set points.Set contains 252 points.
  - Output_u.mat: Grid-X velocity values for each of the training/validation set points.Set contains 252 points.
  - Output_v.mat: Grid-Y velocity values for each of the training/validation set points.Set contains 252 points.
  - Output_p.mat: Grid-Pressure values for each of the training/validation set points.Set contains 252 points.
- Data Files(X):
  - SDF_values.mat: Grid signed distance function to pass shape of airfoil to network.
  - input_re_alpha.mat: Specifies flow variables in the form of reynolds number and angle of attack to the airfoil.
  




