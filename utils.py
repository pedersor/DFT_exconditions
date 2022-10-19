import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_density(r_s):
  return 3 / (4 * np.pi * (r_s**3))


def get_r_s(n):
  return (4 * np.pi * n / 3)**(-1 / 3)


def get_s(n, n_grad):
  if n_grad.ndim == 1:
    n_grad = np.abs(n_grad)
  else:
    n_grad = np.sum(n_grad**2, axis=0)**(1 / 2)
  return n_grad / (2 * ((3 * (np.pi**2) * n)**(1 / 3)) * n)


def get_grad_n(s, n):
  """ Obtain |\nabla n| from the reduced gradient s. """
  return s * (2 * ((3 * np.pi**2)**(1 / 3)) * (n**(4 / 3)))


def use_standard_plotting_params():
  sns.set_theme()
  plt.rcParams["axes.titlesize"] = 24
  plt.rcParams["axes.labelsize"] = 20
  plt.rcParams["lines.linewidth"] = 3
  plt.rcParams["lines.markersize"] = 8
  plt.rcParams["xtick.labelsize"] = 16
  plt.rcParams["ytick.labelsize"] = 16
  plt.rcParams["font.size"] = 24
  plt.rcParams["legend.fontsize"] = 16
  plt.rcParams["figure.figsize"] = (6, 6)