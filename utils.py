import numpy as np


def get_density(r_s):
  return 3 / (4 * np.pi * (r_s**3))


def get_r_s(n):
  return (4 * np.pi * n / 3)**(-1 / 3)


def get_s(n, n_grad):
  if n_grad.ndim == 1:
    n_grad = np.abs(n_grad)
  else:
    n_grad = np.sum(n_grad**2, axis=0)**(1 / 2)
  return n_grad / (2 * (3 * np.pi)**(1 / 3) * n**(4 / 3))


def get_grad_n(s, n):
  """ Obtain |\nabla n| from the reduced gradient s. """
  return s * (2 * ((3 * np.pi**2)**(1 / 3)) * (n**(4 / 3)))
