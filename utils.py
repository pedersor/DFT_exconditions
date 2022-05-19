import numpy as np


def get_r_s(n):
  return (4 * np.pi * n / 3)**(-1 / 3)


def get_s(n, n_grads):
  n_grad = np.sum(n_grads**2, axis=0)**(1 / 2)
  return n_grad / (2 * (3 * np.pi)**(1 / 3) * n**(4 / 3))
