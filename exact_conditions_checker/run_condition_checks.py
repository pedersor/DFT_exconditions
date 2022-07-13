import sys

import numpy as np
import pylibxc

import condition_checks

func_id = sys.argv[1]
condition_string = sys.argv[2]

print(f"running: {func_id}", flush=True)
print(f"checking condition: {condition_string}", flush=True)

func_c = pylibxc.LibXCFunctional(func_id, "polarized")

if 'mgga_c_' in func_id or 'mgga_xc_' in func_id:
  if func_c._needs_laplacian:
    input = {
        'r_s': np.linspace(0.0001, 5, 3000),
        's': np.linspace(0, 5, 100),
        'zeta': np.linspace(0, 1, 20),
        'alpha': np.linspace(0, 5, 10),
        'q': np.linspace(0, 5, 50),
    }
    num_splits = 100
  else:
    input = {
        'r_s': np.linspace(0.0001, 5, 5000),
        's': np.linspace(0, 5, 100),
        'zeta': np.linspace(0, 1, 20),
        'alpha': np.linspace(0, 5, 100),
    }
    num_splits = 50
elif 'gga_c_' in func_id or 'gga_xc_' in func_id:
  input = {
      'r_s': np.linspace(0.0001, 5, 10000),
      's': np.linspace(0, 5, 500),
      'zeta': np.linspace(0, 1, 100),
  }
  num_splits = 100

df = condition_checks.check_condition(
    func_id,
    condition_string,
    input,
    num_splits=num_splits,
)

df.to_csv(f'{func_id}.csv', header=False, index=False)
