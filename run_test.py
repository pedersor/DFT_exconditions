import sys
from pathlib import Path

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
    'r_s_range': [],
    's_range': [],
    'zeta_range': [],
}

range_labels = ['r_s_range', 's_range', 'zeta_range', 'alpha_range', 'q_range']

if 'mgga_c_' in xc:

  df['alpha_range'] = []
  df['q_range'] = []

  if func_c._needs_laplacian:

    r_s = np.linspace(0.001, 2, 10)
    s = np.linspace(0, 5, 10)
    zeta = np.linspace(0, 1, 10)
    alpha = np.linspace(0, 5, 10)
    q = np.linspace(0, 5.0, 50)

    # split up to reduce memory
    r_s_splits = np.split(r_s, 5)
    cond_satisfied = True
    for r_s_split in r_s_splits:

      input = np.meshgrid(r_s_split, s, zeta, alpha, q, indexing='ij')
      eps_c = test_suite.mgga_c_lapl(func_c, *input)
      split_cond_satisfied, ranges = test_suite.deriv_check(input, eps_c)

      del input
      del eps_c

      if not split_cond_satisfied:
        cond_satisfied = False
        for i, r in enumerate(ranges):
          df[range_labels[i]].append(r)

    if cond_satisfied:
      for label in range_labels:
        df[label] = ['---']
    else:
      for label in range_labels:
        min_range = np.amin(df[label])
        max_range = np.amax(df[label])
        df[label] = [[min_range, max_range]]

  else:
    r_s = np.linspace(0.001, 2, 50)
    s = np.linspace(0, 5, 50)
    zeta = np.linspace(0, 1, 50)
    alpha = np.linspace(0, 5, 50)

    # split up to reduce memory
    r_s_splits = np.split(r_s, 5)
    cond_satisfied = True
    for r_s_split in r_s_splits:

      input = np.meshgrid(r_s_split, s, zeta, alpha, indexing='ij')
      eps_c = test_suite.mgga_c(xc, *input)
      split_cond_satisfied, ranges = test_suite.deriv_check(input, eps_c)

      del input
      del eps_c

      if not split_cond_satisfied:
        cond_satisfied = False
        for i, r in enumerate(ranges):
          df[range_labels[i]].append(r)

    if cond_satisfied:
      for label in range_labels:
        df[label] = ['---']
    else:
      for label in range_labels[:-1]:
        min_range = np.amin(df[label])
        max_range = np.amax(df[label])
        df[label] = [[min_range, max_range]]

      # no lapl.
      df[range_labels[-1]] = ['---']

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
