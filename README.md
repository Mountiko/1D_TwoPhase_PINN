# 1D_TwoPhase_PINN
A Physics Informed Neural Network that predicts the a two phase flow in porous media.

Prerequisites:
  - [Julia version 1.3.1](https://julialang.org/)
  - [ADCME](https://github.com/kailaix/ADCME.jl)
  - [FwiFlow](https://github.com/lidongzh/FwiFlow.jl)
  - Python 3.7 +
  
This repository contains the full cycle of training a neural network.
It contains code to do the following 

1. DNN to solve analytical Buckley-Leverett
2. DNN to solve Forward Problem based on simulation data 
3. Calibration tests to solve inverse problem for unknown Permeability and Porosity
4. DNN to solve homogeneous inverse flow
5. DNN for homogeneous approximation of heterogeneous flow + invert permeability and porosity values
6. DNN to solve heterogeneous flow + invert permeability and porosity values


