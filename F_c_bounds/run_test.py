import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pylibxc

import test_suite

xc = sys.argv[1]
cond_to_check = test_suite.deriv_upper_bd_check_1

func_c = pylibxc.LibXCFunctional(xc, "polarized")
print(f"running: {xc}", flush=True)
print(f"checking condition: {cond_to_check.__name__}", flush=True)

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

    r_s = np.linspace(0.0001, 5, 3000)
    s = np.linspace(0, 5, 100)
    zeta = np.linspace(0, 1, 20)
    alpha = np.linspace(0, 5, 10)
    q = np.linspace(0, 5, 50)

    # split up to reduce memory
    s_splits = np.split(s, 100)
    cond_satisfied = True
    for s_split in s_splits:

      input = [r_s, s_split, zeta, alpha, q]
      split_cond_satisfied, ranges = test_suite.check_condition(
          xc,
          cond_to_check,
          input,
      )

      del input

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

    r_s = np.linspace(0.0001, 5, 5000)
    s = np.linspace(0, 5, 100)
    zeta = np.linspace(0, 1, 20)
    alpha = np.linspace(0, 5, 100)

    # split up to reduce memory
    s_splits = np.split(s, 50)
    cond_satisfied = True
    for s_split in s_splits:

      input = [r_s, s_split, zeta, alpha]
      split_cond_satisfied, ranges = test_suite.check_condition(
          xc,
          cond_to_check,
          input,
      )

      del input

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

  range_labels = range_labels[:3]

  r_s = np.linspace(0.0001, 5, 10000)
  s = np.linspace(0, 5, 500)
  zeta = np.linspace(0, 1, 100)

  s_splits = np.split(s, 100)

  cond_satisfied = True
  for s_split in s_splits:
    input = [r_s, s_split, zeta]
    split_cond_satisfied, ranges = test_suite.check_condition(
        xc,
        cond_to_check,
        input,
    )

    del input

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

df['satisfied'] = [cond_satisfied]
df = pd.DataFrame.from_dict(df)
df.to_csv(f'{xc}.csv', header=False, index=False)
