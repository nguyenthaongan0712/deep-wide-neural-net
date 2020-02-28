import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as F
import sys
import scipy
import scipy.linalg

from NTK import *
from util_linnet import *

class DeepArcCosine(object):

    def __init__(self, input_dim, num_steps,
                 variance=1.0, bias_variance=0., active_dims=None):
        self.input_dim =input_dim
        self.num_steps = num_steps
        self.bias_variance=bias_variance
        self.variance=variance
       
    
    def baseK(self, X, X2):
        inner = torch.matmul(X * self.variance, X2.t())/self.input_dim
        return inner + torch.ones_like(inner) * self.bias_variance
        
    def baseKdiag(self, X):
        inner = torch.sum(self.variance*X**2, 1)/self.input_dim
        return inner + torch.ones_like(inner) * self.bias_variance
        
    def K(self, X, X2):
        # initialisation
        K = self.baseK( X, X2 ) 
        kxdiag  = self.baseKdiag(X)
        kx2diag = self.baseKdiag(X2) 
        
        # iterative computation of the kernel
        for step_index in range(self.num_steps):
            K = self.recurseK( K, kxdiag, kx2diag )
            kxdiag = self.recurseKdiag( kxdiag )
            kx2diag = self.recurseKdiag( kx2diag )

        return K

    def recurseK(self, K, kxdiag, kx2diag):
        norms = torch.sqrt(kxdiag)
        norms_rec = torch.rsqrt(kxdiag)
        norms2 = torch.sqrt(kx2diag)
        norms2_rec = torch.rsqrt(kx2diag)
        
        jitter = 1e-7
        scaled_numerator = K * (1.-jitter)

        cos_theta = scaled_numerator * norms_rec[:,None] *  norms2_rec[None,:]   #elementwise multiplication 
        theta = torch.acos(cos_theta) 
        
        return (self.variance / (2*np.pi)) * (torch.sqrt(kxdiag[:,None] * kx2diag[None,:] - torch.mul(scaled_numerator,scaled_numerator) ) + (np.pi - theta) * scaled_numerator ) + self.bias_variance*torch.ones_like(K)
    
    def NTK(self,X,X2=None):
      
        # initialisation
        if X2 is None:
          X2 = X
        NTK = self.baseK( X, X2 )  # because NTK^1 = K^1
        ntkxdiag  = self.baseKdiag(X)
        ntkx2diag = self.baseKdiag(X2) 
        
        K = self.baseK( X, X2 )
        kxdiag  = self.baseKdiag(X)
        kx2diag = self.baseKdiag(X2) 
        
        # recursion
        for step_index in range(self.num_steps):
          # recursively compute relu kernel
          K_new = self.recurseK( K, kxdiag, kx2diag )
          kxdiag_new = self.recurseKdiag( kxdiag )
          kx2diag_new = self.recurseKdiag( kx2diag )
          NTK = self.recurseNTK( NTK, K,kx2diag,kxdiag, K_new)
          K = K_new
          kxdiag = kxdiag_new
          kx2diag = kx2diag_new
           
        return NTK
      
    def recurseNTK(self, NTK,K,kx2diag,kxdiag, K_new):
        norms = torch.sqrt(kxdiag)
        norms_rec = torch.rsqrt(kxdiag)
        norms2 = torch.sqrt(kx2diag)
        norms2_rec = torch.rsqrt(kx2diag)
        
        jitter = 1e-7
        scaled_numerator = K * (1.-jitter)
                
        cos_theta = scaled_numerator * norms_rec[:,None] *  norms2_rec[None,:]
        theta = torch.acos(cos_theta) 
        return K_new + self.variance *(0.5/np.pi)*NTK*(np.pi - theta)
       
      
    def recurseKdiag(self, Kdiag):
        # angle is zero, hence the diagonal stays the same (if relu is used)        
        return 0.5*self.variance * Kdiag  + self.bias_variance * torch.ones_like(Kdiag) 

class NNGP(object):

    def __init__(self, deepArcCosine,X,Y,X_star):
      self.deepArcCosine = deepArcCosine
      self.X = X
      self.Y = Y
      self.X_star = X_star
      self.K = deepArcCosine.K(X,X).cpu().numpy()
      self.K_star = deepArcCosine.K(X_star,X).cpu().numpy()
      self.K_ss = deepArcCosine.K(X_star,X_star).cpu().numpy()
      self.L =  np.linalg.cholesky(self.K+0.00005 * np.eye(X.shape[0]))
      self.n_test = X_star.shape[0]
    
    
    def prior_std(self):
      return np.sqrt(np.diag(self.K_ss))
      
    def posterior_mean(self):
      Lk = np.linalg.solve(self.L, self.K_star.T) 
      mean_star = np.dot(Lk.T, np.linalg.solve(self.L, self.Y.cpu().numpy())).reshape((self.n_test,))
      return mean_star
      
    def posterior_std(self):
      Lk = np.linalg.solve(self.L,self.K_star.T) 
      s2 = np.diag(self.K_ss) - np.sum(Lk**2, axis=0)
      stdv = np.sqrt(s2)
      return(stdv)
      
    def get_K(self):
      return self.K
    
    def get_Kstar(self):
      return self.K_star
    
    def get_Kss(self):
      return self.K_ss

class NTK_analytic_pred(object):

    def __init__(self, deepArcCosine,X,Y,X_star,t,eta,K,K_star,K_ss):
      self.deepArcCosine = deepArcCosine
      self.t=t
      self.eta=eta
      self.X = X
      self.Y = Y
      self.X_star = X_star
      self.K = K
      self.K_star = K_star
      self.K_ss = K_ss
      self.n_test = X_star.shape[0]
      self.NTK_star = deepArcCosine.NTK(X_star,X).cpu().numpy()
      self.NTK_ss = deepArcCosine.NTK(X_star,X_star).cpu().numpy()
      self.NTK = deepArcCosine.NTK(X,X).cpu().numpy()

    def NTK_mean(self):
      L_NTK = np.linalg.cholesky(self.NTK+0.00005 * np.eye(self.X.shape[0]))
      L_help = np.linalg.solve(L_NTK, self.NTK_star.T)  # 4 x 100
      mat = np.eye(4)-scipy.linalg.expm(-self.eta*self.t*self.NTK)
      Y_transf = mat.dot(self.Y.cpu().numpy())
      meanNTK_star = np.matmul(L_help.T, np.linalg.solve(L_NTK, Y_transf)).reshape((self.n_test,))
      return meanNTK_star


    def NTK_std(self):
      inv = np.linalg.inv(self.NTK)
      mat = np.eye(4)-scipy.linalg.expm(-self.eta*self.t*self.NTK)
      Temp = np.matmul(self.NTK_star,np.matmul(inv,mat))
      A = np.matmul(Temp,self.K_star.T)
      B = np.matmul(inv,np.matmul(mat,self.NTK_star.T))
      s2 = np.diag(self.K_ss)-2*np.diag(A)+np.diag(np.matmul(Temp,np.matmul(self.K,B)))
      stdv_NTK = np.sqrt(s2)
      return stdv_NTK
