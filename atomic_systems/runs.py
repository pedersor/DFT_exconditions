import sys
import numpy as np
import pandas as pd

from evaluator import PyscfEvaluator
from dataset import Dataset

#xc = sys.argv[1]
xc = 'm06'

dset = Dataset('ie_atoms.yaml')

scf_args = {'max_cycle': 100, 'conv_tol': 1e-9}
evl = PyscfEvaluator(xc, c=f',{xc}', scf_args=scf_args)

all_sys_checks = []
for i in range(len(dset)):

  curr_calc = dset[i]
  label = curr_calc["name"].split(' ')[-1]

  sys_checks = evl.exact_cond_checks(curr_calc)

  # TODO: remove?
  for sys_check in sys_checks:
    for check in sys_check.values():
      if check is False:
        print()

  all_sys_checks.extend(sys_checks)

flatten_sys_checks = {key: [] for key in all_sys_checks[0].keys()}
for system in all_sys_checks:
  for key in flatten_sys_checks.keys():
    flatten_sys_checks[key].append(system[key])

checks = {key: np.all(value) for key, value in flatten_sys_checks.items()}
df = pd.DataFrame.from_dict(checks)
df.to_csv(f'{xc}.csv', header=False, index=False)
print(checks)