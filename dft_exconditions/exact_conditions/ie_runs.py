import sys
import numpy as np
import pandas as pd

from dft_exconditions.exact_conditions.evaluator import PyscfEvaluator
from dft_exconditions.exact_conditions.dataset import Dataset

DEBUG = False

if DEBUG:
  xc = 'sogga11'
else:
  xc = sys.argv[1]

xc = xc.upper()
if '_XC_' in xc:
  xc_label = xc.split('_XC_')[-1]
elif '_C_' in xc:
  xc_label = xc.split('_C_')[-1]
else:
  xc_label = xc
xc_label = xc_label.replace('_', '-')

dset = Dataset('ie_atoms.yaml')

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

# organize exact condition checks
flatten_sys_checks = {key: [] for key in all_sys_checks[0].keys()}
for system in all_sys_checks:
  for key in flatten_sys_checks.keys():
    flatten_sys_checks[key].append(system[key])

df = pd.DataFrame.from_dict(flatten_sys_checks)
df.to_csv(f'checks_{xc_label}.csv')

checks = {key: np.all(value) for key, value in flatten_sys_checks.items()}
print('checks = ', checks)

# org benchmark errors
df = pd.DataFrame.from_dict(all_sys_errors)
mae = np.mean(np.abs(df['error'].to_numpy())) * dataset.HAR_TO_KCAL
print('MAE = ', mae)
df.to_csv(f'errs_{xc_label}.csv', index=None)
