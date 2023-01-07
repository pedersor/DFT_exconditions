import argparse
from collections import defaultdict

import pandas as pd

from dft_exconditions.exact_conditions.evaluator import PyscfEvaluator
from dft_exconditions.exact_conditions.dataset import Dataset

# if True, only run the first few systems.
DEBUG = True


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--xc",
      type=str,
      help='xc functional to use for calculations',
      default='sogga11',
  )
  xc = parser.parse_args().xc

  xc = xc.upper()
  if '_XC_' in xc:
    xc_label = xc.split('_XC_')[-1]
  elif '_C_' in xc:
    xc_label = xc.split('_C_')[-1]
  else:
    xc_label = xc
  xc_label = xc_label.replace('_', '-')

  dset = Dataset('ea_atoms.yaml')

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
    # evaluate on HF densities
    evl = PyscfEvaluator(xc=None, hf=True, scf_args=scf_args)

    curr_calc = dset[i]
    label = curr_calc["name"].split(' ')[-1]
    all_sys_errors['System'].append(label)

    evl.xc = xc
    # benchmark errors
    sys_error = evl.get_error(curr_calc)
    all_sys_errors[xc_label].append(sys_error)

    # exact condition checks
    sys_checks = evl.get_exact_cond_checks(curr_calc)

    if not all_sys_checks.get(xc_label, False):
      # initialize for XC
      cond_labels = [*sys_checks[0]]
      all_sys_checks[xc_label] = {cond: [] for cond in cond_labels}

    for syst in sys_checks:
      for cond, val in syst.items():
        all_sys_checks[xc_label][cond].append(val)

    # remove cached mf objects
    evl.reset_mfs()

    if DEBUG and i == 2:
      break

  # organize exact condition checks as separate csv files for each XC.
  cond_checks_out_file = f'ea_checks_{xc}.csv'
  for xc, checks in all_sys_checks.items():
    df = pd.DataFrame.from_dict(checks)
    df.to_csv(cond_checks_out_file)
  print('output exact condition checks to file: ', cond_checks_out_file)

  # benchmark errors
  errs_out_file = f'ea_errs_{xc_label}.csv'
  df = pd.DataFrame.from_dict(all_sys_errors)
  df.to_csv(errs_out_file, index=None)
  print('output abs. energy errors to file: ', errs_out_file)


if __name__ == '__main__':
  main()