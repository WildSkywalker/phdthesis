import torch
from torch import nn
from torch import functional as F
import numpy as np
import scipy as sp
import math

print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

def pairwise_distances(X):
  """
  Computes pairwise squared Euclidean distances between data points.

  Args:
    X: A PyTorch tensor of shape (n_samples, n_features)

  Returns:
    A PyTorch tensor of shape (n_samples, n_samples) containing the pairwise
    distances.
  """
  X_sum = torch.sum(X ** 2, dim=1, keepdim=True)
  dist = X_sum + X_sum.T - 2 * torch.mm(X, X.T)
  return dist

def gaussian_kernel(distances, sigma):
  """
  Computes the Gaussian kernel matrix.

  Args:
    distances: A PyTorch tensor of shape (n_samples, n_samples) containing the
      pairwise distances.
    sigma: A PyTorch tensor of shape (n_samples,) containing the bandwidth for
      each data point.

  Returns:
    A PyTorch tensor of shape (n_samples, n_samples) containing the kernel matrix.
  """
  # Prevent division by zero
  sigma = sigma.unsqueeze(1) + 1e-8 
  exponent = -distances / (2 * (sigma ** 2))
  return torch.exp(exponent)