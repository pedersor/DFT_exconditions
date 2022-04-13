import sys

import pylibxc
import numpy as np
import matplotlib.pyplot as plt

# pandas used for convience for data display
import pandas as pd

# start defintions ====


def get_density(r_s):
  return 3 / (4 * np.pi * (r_s**3))


def hartree_to_mRy(energy):
  return energy * 2 * 1000


def get_eps_x_unif(n):
  return -(3 / (4 * np.pi)) * ((n * 3 * np.pi**2)**(1 / 3))


def get_grad_n(s, n):
  return s * (2 * ((3 * np.pi**2)**(1 / 3)) * (n**(4 / 3)))


def get_up_dn_density(n, zeta):
  n = np.expand_dims(n.flatten(), axis=1)
  zeta = np.expand_dims(zeta.flatten(), axis=1)

  up_density = ((zeta * n) + n) / 2
  dn_density = (n - (zeta * n)) / 2
  return np.concatenate((up_density, dn_density), axis=1)


def get_tau(alpha, grad_n, n, zeta):
  tau_w = (grad_n**2) / (8 * n)
  tau_unif = (3 / 10) * ((3 * np.pi)**(2 / 3)) * (n**(5 / 3))
  d_s = ((1 + zeta)**(5 / 3) + (1 - zeta)**(5 / 3)) / 2
  tau_unif *= d_s

  return (alpha * tau_unif) + tau_w


# end defintions ====


def scan_eps_c_pol(r_s, zeta, s, alpha, config):
  """ Gets eps_c^SCAN from (r_s, zeta, s, alpha). config is used
  to specify some example libxc input cases (see cases below). """

  scan_c = pylibxc.LibXCFunctional("mgga_c_scan", "polarized")

  # obtain libxc inputs
  n = get_density(r_s)
  grad_n = get_grad_n(s, n)
  n_spin = get_up_dn_density(n, zeta)
  tau = get_tau(alpha, grad_n, n, zeta)

  # total sigma and tau
  sigma = np.expand_dims(grad_n.flatten()**2, axis=1)
  tau = np.expand_dims(tau.flatten(), axis=1)

  if config == 0:
    sigma = np.concatenate((sigma, np.zeros_like(sigma), np.zeros_like(sigma)),
                           axis=1)
    tau = np.concatenate((tau, np.zeros_like(tau)), axis=1)
  elif config == 1:
    sigma = np.concatenate((np.zeros_like(sigma), np.zeros_like(sigma), sigma),
                           axis=1)
    tau = np.concatenate((np.zeros_like(tau), tau), axis=1)
  elif config == 2:
    sigma = np.concatenate((sigma / 4, sigma / 4, sigma / 4), axis=1)
    tau = np.concatenate((tau / 2, tau / 2), axis=1)

  inp = {}
  inp["rho"] = n_spin
  inp["sigma"] = sigma
  inp["tau"] = tau

  scan_c_res = scan_c.compute(inp)
  return np.squeeze(scan_c_res['zk'])


def scan_F_x_unpol(r_s, s, alpha):
  """ Gets eps_x^SCAN from (r_s, zeta, s, alpha). config is used
  to specify some example libxc input cases (see cases below). """

  scan_x = pylibxc.LibXCFunctional("mgga_x_scan", "unpolarized")

  m_r_s, m_s, m_alpha = np.meshgrid(r_s, s, alpha, indexing='ij')
  m_n = get_density(m_r_s)
  m_eps_x_unif = get_eps_x_unif(m_n)

  # obtain libxc inputs
  m_n = get_density(m_r_s)
  m_grad_n = get_grad_n(m_s, m_n)
  m_eps_x_unif = get_eps_x_unif(m_n)
  m_tau = get_tau(m_alpha, m_grad_n, m_n, zeta=0)

  inp = {}
  inp["rho"] = m_n.flatten()
  inp["sigma"] = m_grad_n.flatten()**2
  inp["tau"] = m_tau.flatten()

  scan_x_res = scan_x.compute(inp)
  scan_eps_x = np.squeeze(scan_x_res['zk'])
  scan_F_x = scan_eps_x.reshape(m_r_s.shape) / m_eps_x_unif

  return scan_F_x


def scan_eps_c_unpol(r_s, s, alpha):
  """ Gets eps_c^SCAN from (r_s, zeta, s, alpha). config is used
  to specify some example libxc input cases (see cases below). """

  scan_c = pylibxc.LibXCFunctional("mgga_c_scan", "unpolarized")

  # obtain libxc inputs
  n = get_density(r_s)
  grad_n = get_grad_n(s, n)
  tau = get_tau(alpha, grad_n, n, zeta=0)

  # total sigma and tau
  sigma = np.expand_dims(grad_n.flatten()**2, axis=1)
  tau = np.expand_dims(tau.flatten(), axis=1)

  inp = {}
  inp["rho"] = n
  inp["sigma"] = sigma
  inp["tau"] = tau

  scan_c_res = scan_c.compute(inp)
  return np.squeeze(scan_c_res['zk'])


def pbe_eps_c_pol(r_s, zeta, s, config):
  """ Gets eps_c^PBE from (r_s, zeta, s). config is used
  to specify some example libxc input cases (see cases below). """

  pbe_c = pylibxc.LibXCFunctional("gga_c_pbe", "polarized")

  # obtain libxc inputs
  n = get_density(r_s)
  grad_n = get_grad_n(s, n)
  n_spin = get_up_dn_density(n, zeta)

  # total sigma and tau
  sigma = np.expand_dims(grad_n.flatten()**2, axis=1)

  if config == 0:
    sigma = np.concatenate((sigma, np.zeros_like(sigma), np.zeros_like(sigma)),
                           axis=1)
  elif config == 1:
    sigma = np.concatenate((np.zeros_like(sigma), np.zeros_like(sigma), sigma),
                           axis=1)
  elif config == 2:
    sigma = np.concatenate((sigma / 4, sigma / 4, sigma / 4), axis=1)

  inp = {}
  inp["rho"] = n_spin
  inp["sigma"] = sigma

  pbe_c_res = pbe_c.compute(inp)
  return np.squeeze(pbe_c_res['zk'])


example = 'scan_Fx_simple'

if example == 'scan_Fx_simple':
  n = 1
  tau_unif = (3 / 10) * ((3 * np.pi)**(2 / 3)) * (n**(5 / 3))
  eps_x_unif = -(3 / (4 * np.pi)) * ((n * 3 * np.pi**2)**(1 / 3))

  scan_x = pylibxc.LibXCFunctional("mgga_x_scan", "unpolarized")

  inp = {}
  inp["rho"] = [n]
  inp["sigma"] = [0]
  inp["tau"] = [tau_unif]

  scan_x_res = scan_x.compute(inp)
  scan_eps_x = np.squeeze(scan_x_res['zk'])
  scan_F_x = scan_eps_x / eps_x_unif

  print(scan_F_x)

if example == 'scan_Fx':

  r_s = np.array([1])
  s = np.linspace(0, 3)
  alpha = np.array([0, 1, 100])

  f_x = scan_F_x_unpol(r_s, s, alpha)

  # remove dummy r_s dim
  f_x = np.squeeze(f_x)
  # swap s and alpha axes for plotting
  f_x = np.transpose(f_x, [1, 0])

  for a, f_x_plot in zip(alpha, f_x):
    plt.plot(s, f_x_plot, label=rf'$\alpha = {a}$')

  plt.ylim(bottom=0.75, top=1.5)
  plt.xlim(left=0, right=3)
  plt.legend()
  plt.xlabel("$s$")
  plt.ylabel(r"$F_x(S, \alpha)$")
  plt.grid(alpha=0.4)
  plt.savefig(f'{example}.pdf')

  print()

if example == "scan eps_c":
  """ example. """

  r_s = np.array([2])
  s = np.array([2])
  alpha = np.array([2])
  zetas = [0, 1, -1]
  configs = [0, 1, 2]

  df = {zeta: [] for zeta in zetas}
  for config in configs:
    for zeta in df:
      df[zeta].append(
          scan_eps_c_pol(
              r_s=r_s,
              zeta=np.array([zeta]),
              s=s,
              alpha=alpha,
              config=config,
          ))

  df["config"] = configs

  # if don't have pandas, can just print python dict:
  #print(df)

  df = pd.DataFrame.from_dict(df)
  col_names = [f"zeta = {zeta}" for zeta in zetas]
  df.rename(columns={zeta: col for zeta, col in zip(zetas, col_names)},
            inplace=True)
  print('libxc result:')
  print(df.to_string(index=False))

  #print("""\n values from my mathematica implementation:
  #  zeta=0 : -0.0194047
  #  zeta=1,-1 : -0.010152 """)
