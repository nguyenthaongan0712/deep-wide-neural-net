import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim, nn
import copy
import torch.nn.functional as F
import sys
import scipy
import scipy.linalg

from init import *
from NTK import *
from util_linnet import *
from NNGP_vs_NTK import *

def analytical_evolution_MSE(t, lr, theta_0, theta_0_test, initial_train, initial_test, target_data, grad_tensor):
  # compute the analytical evolution of the weights using the equation 9 in the linearized network paper,
  # and the prediction on the train and test set using equations 10, 11, 12. 
    
  # This is valid for MSE loss. 
  
  # t -> time after you want to consider the weights
  # lr -> learning rate
  # theta_0 -> NTK computed at initialization on the train set
  # theta_0_test -> NTK computed at initialization on the test set vs the train set
  # initial_train -> prediction using the initial value of the network on the train set. 
  # initial_test -> prediction using the initial value of the network on the test set.
  # target data -> ground truth of the training points
  # grad_tensor -> tensor containing the gradients with respect to the parameters 
  # of the network that you want to consider, computed in all the training points. 
  # Shape `(n_train_points, n_parameters)`

  n_train = len(initial_train)
  
  # first compute the exponential of the matrix (using eigendecomposition): 
  lam, P = np.linalg.eig(theta_0)  # eig decomposition
  lam = lam.astype(dtype = 'float64')
  
  theta_0_inv = np.dot(P, np.dot(np.diag(lam**(-1)), P.transpose())) 
  
  # note that you need to rescale the time by n_train, as the 2 paper use different convention for the loss function
  exp_matrix = np.dot(P, np.dot(np.diag(np.exp(-lr * t * lam / n_train)), P.transpose()))  # I am using np arrays, not torch tensors
    
  # compute the prediction on train set
  pred_train = target_data.cpu().numpy() + np.dot(exp_matrix, (initial_train - target_data).cpu().detach().numpy())
  
  # compute the intermediate matrix used both in prediction on test set and weights evolution
  tmp = np.dot(np.eye(lam.size) - exp_matrix, (initial_train - target_data).cpu().detach().numpy())
  tmp = np.dot(theta_0_inv, tmp)
  
  # compute prediction on test set
  pred_test = np.dot(theta_0_test , tmp)
  pred_test = initial_test.detach().cpu().numpy().reshape(-1) - pred_test
  
  # compute evolution of the weight changes
  weights = - np.dot(grad_tensor.transpose(1,0), tmp)

  return(weights, pred_train, pred_test)
