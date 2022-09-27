import sys
import numpy as np
import pandas as pd
from collections import defaultdict

from evaluator import PyscfEvaluator
import dataset
from dataset import Dataset

DEBUG = False

if DEBUG:
  xcs = {'pbe': 'PBE'}

dset = Dataset('ea_atoms.yaml')

xcs = {
    'm06': 'M06',
    'pbe': 'PBE',
    'sogga11': 'SOGGA11',
    'scan': 'SCAN',
    'r2scan': 'R2SCAN',
    'b3lyp': 'B3LYP',
    'mn15': 'MN15',
    'HYB_GGA_XC_B97': 'B97',
    'gga_x_am05,gga_c_am05': 'AM05',
    #'HYB_GGA_XC_CASE21',
}

scf_args = {
    'max_cycle': 200,
    'conv_tol': 1e-6,
    'diis_space': 12,
    'chkfile': False,
    'verbose': 4,
}

all_sys_checks = defaultdict(list)
all_sys_errors = defaultdict(list)
for i in range(len(dset)):
  print('i = ', i)
  evl = PyscfEvaluator(xc=None, hf=True, scf_args=scf_args)

  curr_calc = dset[i]
  label = curr_calc["name"].split(' ')[-1]
  all_sys_errors['System'].append(label)
  for xc, pretty_xc in xcs.items():
    print('xc = ', xc)
    evl.xc = xc
    # benchmark errors
    sys_error = evl.get_error(curr_calc)
    all_sys_errors[pretty_xc].append(sys_error)

    # exact condition checks
    sys_checks = evl.get_exact_cond_checks(curr_calc)

    if not all_sys_checks.get(pretty_xc, False):
      # initialize for XC
      cond_labels = [*sys_checks[0]]
      all_sys_checks[pretty_xc] = {cond: [] for cond in cond_labels}

    for syst in sys_checks:
      for cond, val in syst.items():
        all_sys_checks[pretty_xc][cond].append(val)

  # remove cached mf objects
  evl.reset_mfs()

  if DEBUG and i == 2:
    break

# benchmark errors
df = pd.DataFrame.from_dict(all_sys_errors)
df.to_csv(f'ea_errs.csv', index=None)

# organize exact condition checks as separate csv files for each XC.
for xc, checks in all_sys_checks.items():
  df = pd.DataFrame.from_dict(checks)
  df.to_csv(f'ea_checks_{xc}.csv')
