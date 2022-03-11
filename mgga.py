import sys

import pylibxc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import utils

eps_x = eps_c = 'zk'


def get_min_i(functional_id, r_s, s, alpha):

  fun_c = pylibxc.LibXCFunctional(functional_id, "polarized")

  # todo figure out mGGA zeta issue...
  zetas = [np.array([0]), np.array([1])]

  min_i = []
  for zeta in zetas:

    m_r_s, m_zeta, m_s, m_alpha = np.meshgrid(
        r_s,
        zeta,
        s,
        alpha,
        indexing='ij',
    )

    m_n = utils.get_density(m_r_s)
    m_grad_n = utils.get_grad_n(m_s, m_n)
    m_eps_x_unif = utils.get_eps_x_unif(m_n)
    m_n_spin = utils.get_up_dn_density(m_n, m_zeta)
    m_tau = utils.get_tau(m_alpha, m_grad_n, m_n)

    # create input
    inp = {}

    # density
    inp["rho"] = m_n_spin

    # (| \nabla n |^2, 0, 0) to follow libxc convention
    sigma = np.expand_dims(m_grad_n.flatten()**2, axis=1)

    tau = np.expand_dims(m_tau.flatten(), axis=1)

    if zeta[0] == 0:
      sigma = np.concatenate((sigma / 4, sigma / 4, sigma / 4), axis=1)
      tau = np.concatenate((tau / 2, tau / 2), axis=1)
    elif zeta[0] == 1:
      sigma = np.concatenate(
          (sigma, np.zeros_like(sigma), np.zeros_like(sigma)), axis=1)

      tau = np.concatenate((tau, np.zeros_like(tau)), axis=1)
    else:
      print('error not implemented yet')

    inp["sigma"] = sigma
    inp["tau"] = tau

    # results in a.u.
    f_c = fun_c.compute(inp)

    f_c = np.squeeze(f_c[eps_c])
    f_c = f_c.reshape(m_r_s.shape)
    f_c /= m_eps_x_unif

    # check all results
    # F_c(r_s', ...) > F_c(r_s, ...) for r_s' > r_s

    min_diff = np.amin(np.diff(f_c, axis=0))
    min_i.append(min_diff)

  return min(min_i)


alpha = np.linspace(0.01, 100, num=500)
r_s = np.linspace(0.1, 100, num=500)
s = np.linspace(0, 5)

functionals = ['scan', 'rscan', 'r2scan', 'revscan', 'tpss', 'm06', 'b88']
functionals = ['mgga_c_' + fun for fun in functionals]

df = {'functional': [], 'min_i': [], 'sign': []}
for fun in functionals:

  res = get_min_i(fun, r_s, s, alpha)

  if res > 0:
    sign = '+'
  else:
    sign = '-'

  df['functional'].append(fun.replace('mgga_c_', '').upper())
  df['min_i'].append(res)
  df['sign'].append(sign)

pd.set_option('display.float_format', '{:.2E}'.format)
df = pd.DataFrame.from_dict(df)
latex_code = df.to_latex(index=False, escape=False)
latex_code = latex_code.replace("\\\n", "\\ \hline\n")
print(latex_code)
