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
      default='gga_c_sogga11',
  )
  parser.add_argument(
      "-c",
      "--condition_string",
      help="The condition string to check.",
      type=str,
      default='negativity_check',
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

  inp = local_condition_checks.default_input_grid_search(func_id)

  df = local_condition_checks.check_condition(
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
