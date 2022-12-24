import textwrap
import subprocess
from pathlib import Path
from datetime import datetime
import shutil

import pylibxc

DEBUG = False

func_types = [
    'gga_c_',
    'gga_xc_',
    'hyb_gga_xc_',
    'mgga_c_',
    'mgga_xc_',
    'hyb_mgga_xc_',
]

conditions_to_check = [
    "negativity_check",
    "deriv_lower_bd_check",
    "deriv_upper_bd_check_1",
    "deriv_upper_bd_check_2",
    "second_deriv_check",
    "lieb_oxford_bd_check_Uxc",
    "lieb_oxford_bd_check_Exc",
]

range_type = 'comprehensive'

# file that we will run on the cluster
run_file = Path('run_condition_checks.py').resolve()
# output directory with date and time
date_time = datetime.now().strftime("%m-%d-%y_%H-%M")
out_dir = Path('out_' + date_time)
out_dir.mkdir(parents=True, exist_ok=True)
# temporary files during run
tmp_dir = out_dir / Path('tmp_files')
tmp_dir.mkdir(parents=True, exist_ok=True)
# files to organize data
shutil.copy2('organize_data.py', out_dir)
# copy over specific files and parameters used in the run for reference.
# These files should be able to reproduce the run results.
run_files = out_dir / 'run_files'
run_files.mkdir(parents=True, exist_ok=True)
shutil.copy2(run_file, run_files)
shutil.copy2(Path(__file__).resolve(), run_files)

# all available XC functionals in libxc
xc_funcs = pylibxc.util.xc_available_functional_names()
for func_type in func_types:
  for xc_func in xc_funcs:
    # filter out 1d and 2d functionals
    if not (func_type in xc_func and func_type[0] == xc_func[0] and
            '_1d_' not in xc_func and '_2d_' not in xc_func):
      continue

    for cond_to_check in conditions_to_check:

      run_cmd = f'python3 {run_file} {xc_func} {cond_to_check} {range_type} '
      job_file = f'{xc_func}_{cond_to_check}.job'
      with open(tmp_dir / job_file, "w") as fh:
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
              srun {run_cmd}

          """)

        lines = textwrap.dedent(lines)
        fh.writelines(lines)

      if DEBUG:
        cmd = run_cmd
      else:
        cmd = f"sbatch {job_file}"
      # slurm batch submit
      proc = subprocess.Popen(
          cmd,
          shell=True,
          cwd=tmp_dir,
          stdout=None,
      )
      proc.wait()
