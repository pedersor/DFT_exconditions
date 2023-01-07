import argparse
import numpy as np
import pandas as pd

from dft_exconditions.evaluator import PyscfEvaluator
from dft_exconditions import dataset

# if DEBUG is True, only run the first few systems.
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

  dset = dataset.Dataset('ie_atoms.yaml')

  scf_args = {
      'max_cycle': 200,
      'conv_tol': 1e-6,
      'diis_space': 12,
      'chkfile': False,
      'verbose': 4,
  }
  evl = PyscfEvaluator(xc, scf_args=scf_args)

  all_sys_checks = []
  all_sys_errors = {'label': [], 'error': []}
  for i in range(len(dset)):

    curr_calc = dset[i]
    label = curr_calc["name"].split(' ')[-1]

    # exact condition checks
    sys_checks = evl.get_exact_cond_checks(curr_calc)
    all_sys_checks.extend(sys_checks)

    # benchmark errors
    sys_error = evl.get_error(curr_calc)
    all_sys_errors['label'].append(label)
    all_sys_errors['error'].append(sys_error)

    # remove cached mf objects
    evl.reset_mfs()

    if DEBUG and i == 3:
      break

  # organize exact condition checks in csv file
  cond_checks_out_file = f'ie_checks_{xc_label}.csv'

  flatten_sys_checks = {key: [] for key in all_sys_checks[0].keys()}
  for system in all_sys_checks:
    for key in flatten_sys_checks.keys():
      flatten_sys_checks[key].append(system[key])

  df = pd.DataFrame.from_dict(flatten_sys_checks)
  df.to_csv(cond_checks_out_file)
  print('output exact condition checks to file: ', cond_checks_out_file)

  # organize benchmark errors in csv file
  errs_out_file = f'ie_errs_{xc_label}.csv'
  df = pd.DataFrame.from_dict(all_sys_errors)
  df.to_csv(errs_out_file, index=None)
  print('output abs. energy errors to file: ', errs_out_file)


if __name__ == '__main__':
  main()