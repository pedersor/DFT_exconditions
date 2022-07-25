import textwrap
import subprocess
from pathlib import Path

import pylibxc

func_types = ['gga_xc_', 'mgga_xc_', 'mgga_c_']
cond_to_check = 'lieb_oxford_bd_check_Exc'

xc_funcs = pylibxc.util.xc_available_functional_names()

# out directory
out_dir = Path(cond_to_check)
out_dir.mkdir(parents=True, exist_ok=True)

for func_type in func_types:
  for xc_func in xc_funcs:
    # filter out 1d and 2d functionals
    if func_type in xc_func and func_type[0] == xc_func[
        0] and '_1d_' not in xc_func and '_2d_' not in xc_func:

      run_file = '../run_condition_checks.py'
      job_file = f'{xc_func}.job'
      with open(out_dir / job_file, "w") as fh:
        # slurm commands
        lines = (f"""\
            #!/bin/bash
            #SBATCH --job-name="{xc_func}"
            #SBATCH --account=burke
            #SBATCH --partition=sib2.9,nes2.8
            #SBATCH --ntasks=1
            #SBATCH --nodes=1
            #SBATCH --cpus-per-task=8
            #SBATCH --time=10:00:00
            #SBATCH --mem=32G

            ml purge
            ml miniconda/3/own
            srun python {run_file} {xc_func} {cond_to_check}

        """)

        lines = textwrap.dedent(lines)
        fh.writelines(lines)

      # slurm batch submit
      proc = subprocess.Popen(
          f"sbatch {job_file}",
          shell=True,
          cwd=out_dir,
          stdout=None,
      )
