import argparse
from pathlib import Path

from dft_exconditions import local_condition_checks


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-f",
      "--func_id",
      help="The Libxc (X)C functional id to check.",
      type=str,
      default='gga_c_pbe',
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
  num_blocks = args.num_blocks
  output_dir = Path(args.output_dir)

  print(f"running: {func_id}", flush=True)

  f = local_condition_checks.Functional(func_id)
  checker = local_condition_checks.LocalCondChecker(f)
  df = checker.check_conditions(num_blocks=20)

  output_dir.mkdir(parents=True, exist_ok=True)
  df.to_csv(
      output_dir / f'{func_id}.csv',
      header=True,
      index=False,
  )


if __name__ == "__main__":
  main()
