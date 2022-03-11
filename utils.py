import numpy as np


def get_density(r_s):
  return 3 / (4 * np.pi * (r_s**3))


def hartree_to_mRy(energy):
  return energy * 2 * 1000


def get_eps_x_unif(n):
  return -(3 / (4 * np.pi)) * ((n * 3 * np.pi**2)**(1 / 3))


def get_grad_n(s, n):
  return s * ((n**(4 / 3)) * 2 * (3 * np.pi**2)**(1 / 3))


def get_up_dn_density(n, zeta):
  n = np.expand_dims(n.flatten(), axis=1)
  zeta = np.expand_dims(zeta.flatten(), axis=1)

  up_density = ((zeta * n) + n) / 2
  dn_density = (n - (zeta * n)) / 2
  return np.concatenate((up_density, dn_density), axis=1)


def get_tau(alpha, grad_n, n):
  tau_w = (grad_n**2) / (8 * n)
  tau_unif = (3 / 10) * ((3 * np.pi)**(2 / 3)) * (n**(5 / 3))

  return alpha * tau_unif + tau_w
