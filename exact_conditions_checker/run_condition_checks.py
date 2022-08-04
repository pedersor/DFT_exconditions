import sys

import numpy as np
import pylibxc

import condition_checks

func_id = sys.argv[1]
condition_string = sys.argv[2]
range_type = sys.argv[3]

print(f"running: {func_id}", flush=True)
print(f"checking condition: {condition_string}", flush=True)
print(f"range type: {range_type}", flush=True)

func_c = pylibxc.LibXCFunctional(func_id, "polarized")

if range_type == 'comprehensive':
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
elif range_type == 'unpol_nonzero_s':
  if 'mgga_c_' in func_id or 'mgga_xc_' in func_id:
    if func_c._needs_laplacian:
      input = {
          'r_s': np.linspace(0.01, 5, 3000),
          's': np.linspace(0.1, 5, 100),
          'zeta': np.array([0]),
          'alpha': np.linspace(0, 5, 10),
          'q': np.linspace(0, 5, 50),
      }
      num_splits = 100
    else:
      input = {
          'r_s': np.linspace(0.01, 5, 5000),
          's': np.linspace(0.1, 5, 100),
          'zeta': np.array([0]),
          'alpha': np.linspace(0, 5, 100),
      }
      num_splits = 50
  elif 'gga_c_' in func_id or 'gga_xc_' in func_id:
    input = {
        'r_s': np.linspace(0.01, 5, 10000),
        's': np.linspace(0.1, 5, 500),
        'zeta': np.array([0]),
    }
    num_splits = 100

  df = condition_checks.check_condition(
      func_id,
      condition_string,
      input,
      num_splits=num_splits,
  )

elif range_type == 'unpol_small_s':
  if 'mgga_c_' in func_id or 'mgga_xc_' in func_id:
    if func_c._needs_laplacian:
      input = {
          'r_s': np.linspace(0.01, 5, 3000),
          's': np.linspace(0, 0.1, 10),
          'zeta': np.array([0]),
          'alpha': np.linspace(0, 5, 10),
          'q': np.linspace(0, 5, 50),
      }
      num_splits = 5
    else:
      input = {
          'r_s': np.linspace(0.01, 5, 5000),
          's': np.linspace(0, 0.1, 10),
          'zeta': np.array([0]),
          'alpha': np.linspace(0, 5, 100),
      }
      num_splits = 5
  elif 'gga_c_' in func_id or 'gga_xc_' in func_id:
    input = {
        'r_s': np.linspace(0.01, 5, 10000),
        's': np.linspace(0, 0.1, 10),
        'zeta': np.array([0]),
    }
    num_splits = 5

  df = condition_checks.check_condition(
      func_id,
      condition_string,
      input,
      num_splits=num_splits,
  )
else:
  NotImplementedError(f"range_type {range_type} not supported.")

df.to_csv(f'{func_id}.csv', header=False, index=False)
