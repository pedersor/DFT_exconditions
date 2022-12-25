import argparse
from pathlib import Path

import numpy as np
import pylibxc

from dft_exconditions.local_conditions import condition_checks


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-f",
      "--func_id",
      help="The Libxc (X)C functional id to check.",
      type=str,
  )
  parser.add_argument(
      "-c",
      "--condition_string",
      help="The condition string to check.",
      type=str,
  )
  parser.add_argument(
      "-n",
      "--num_blocks",
      help="The number of blocks to split the grid search into.",
      type=int,
      default=100,
  )
  parser.add_argument(
      "-o",
      "--output_dir",
      help=("Dir to store output results. If no dir exists, it will "
            "be created. Default: out/ ."),
      type=str,
      default='out/',
  )
  args = parser.parse_args()

  func_id = args.func_id
  condition_string = args.condition_string
  num_blocks = args.num_blocks
  output_dir = Path(args.output_dir)

  print(f"running: {func_id}", flush=True)
  print(f"checking condition: {condition_string}", flush=True)

  libxc_fun = pylibxc.LibXCFunctional(func_id, "polarized")

  if 'mgga_c_' in func_id or 'mgga_xc_' in func_id:
    if libxc_fun._needs_laplacian:
      inp = {
          'r_s': np.linspace(0.0001, 5, 3000),
          's': np.linspace(0, 5, 100),
          'zeta': np.linspace(0, 1, 20),
          'alpha': np.linspace(0, 5, 10),
          'q': np.linspace(0, 5, 50),
      }
    else:
      inp = {
          'r_s': np.linspace(0.0001, 5, 5000),
          's': np.linspace(0, 5, 100),
          'zeta': np.linspace(0, 1, 20),
          'alpha': np.linspace(0, 5, 100),
      }
  elif 'gga_c_' in func_id or 'gga_xc_' in func_id:
    inp = {
        'r_s': np.linspace(0.0001, 5, 500),
        's': np.linspace(0, 5, 500),
        'zeta': np.linspace(0, 1, 100),
    }

  df = condition_checks.check_condition(
      func_id,
      condition_string,
      inp,
      num_blocks=num_blocks,
  )

  output_dir.mkdir(parents=True, exist_ok=True)
  df.to_csv(
      output_dir / f'{func_id}_{condition_string}.csv',
      header=True,
      index=False,
  )


if __name__ == "__main__":
  main()
