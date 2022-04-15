import sys

import pandas as pd
import numpy as np
import pylibxc

import test_suite

#xc = sys.argv[1]
xc = 'mgga_c_m06'
func_c = pylibxc.LibXCFunctional(xc, "polarized")

df = {
    'xc': [xc],
    'satisfied': [],
}

range_labels = ['r_s_range', 's_range', 'zeta_range', 'alpha_range', 'q_range']

if 'mgga_c_' in xc:

  r_s = np.linspace(0.001, 2, 50)
  s = np.linspace(0, 5, 50)
  zeta = np.linspace(0, 1, 50)
  alpha = np.linspace(0, 5, 50)

  if func_c._needs_laplacian:

    # TODO remove
    r_s = np.linspace(0.001, 2, 10)
    s = np.linspace(0, 5, 10)
    zeta = np.linspace(0, 1, 10)
    alpha = np.linspace(0, 5, 10)

    q = np.linspace(0, 5.0, 50)
    input = np.meshgrid(r_s, s, zeta, alpha, q, indexing='ij')
    eps_c = test_suite.mgga_c_lapl(func_c, *input)
    cond_satisfied, ranges = test_suite.deriv_check(input, eps_c)

    if ranges is not None:
      for i, r in enumerate(ranges):
        df[range_labels[i]] = [r]

    else:
      for label in range_labels:
        df[label] = ['---']

  else:
    input = np.meshgrid(r_s, s, zeta, alpha, indexing='ij')
    eps_c = test_suite.mgga_c(xc, *input)
    cond_satisfied, ranges = test_suite.deriv_check(input, eps_c)

    if ranges is not None:
      for i, r in enumerate(ranges):
        df[range_labels[i]] = [r]

      # no lapl.
      df[range_labels[-1]] = ['---']

    else:
      for label in range_labels:
        df[label] = ['---']

elif 'gga_c_' in xc:

  r_s = np.linspace(0.001, 2, 50)
  s = np.linspace(0, 5, 50)
  zeta = np.linspace(0, 1, 50)
  input = np.meshgrid(r_s, s, zeta, indexing='ij')

  eps_c = test_suite.gga_c(xc, *input)

  cond_satisfied, ranges = test_suite.deriv_check(input, eps_c)

  if ranges is not None:
    for i, r in enumerate(ranges):
      df[range_labels[i]] = [r]

  else:
    for label in range_labels[:3]:
      df[label] = ['---']

df['satisfied'] = [cond_satisfied]
df = pd.DataFrame.from_dict(df)
df.to_csv(f'{xc}.csv', header=False, index=False)
