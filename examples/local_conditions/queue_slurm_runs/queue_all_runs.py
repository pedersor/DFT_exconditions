import textwrap
import subprocess
from pathlib import Path
import shutil
from datetime import datetime

import pylibxc
import yaml

DEBUG = True

func_types = [
    'lda_c_',
    'lda_xc_',
    'hyb_lda_xc_',
    'gga_c_',
    'gga_xc_',
    'hyb_gga_xc_',
    'mgga_c_',
    'mgga_xc_',
    'hyb_mgga_xc_',
]

# output directory with date and time
date_time = datetime.now().strftime("%m-%d-%y_%H-%M")
out_dir = Path('out_' + date_time)
out_dir.mkdir(parents=True, exist_ok=True)

# slurm config
with open('slurm_config.yaml', 'r') as file:
  slurm_config = yaml.safe_load(file)

  load_python_cmd = slurm_config['load_python_cmd']
  # separate commands by line breaks
  if isinstance(load_python_cmd, list):
    load_python_cmd = '\n'.join(load_python_cmd)

# copy over specific files and parameters used in the run for reference.
# These files should be able to reproduce the run results.
run_file = Path('../local_conditions_runs.py')
run_files = [
    run_file,
    Path('organize_data.py'),
    # other relevant files
]
for file in run_files:
  shutil.copy2(file, out_dir)

# all available XC functionals in libxc
xc_funcs = pylibxc.util.xc_available_functional_names()
for func_type in func_types:
  for xc_func in xc_funcs:
    # filter out 1d and 2d functionals
    if not (func_type in xc_func and func_type[0] == xc_func[0] and
            '_1d_' not in xc_func and '_2d_' not in xc_func):
      continue

    job_file = f'{xc_func}.job'
    run_cmd = f'srun python {run_file.name} -f {xc_func} '
    debug_cmd = f'python3 {run_file.name} -f {xc_func}'
    with open(out_dir / job_file, "w") as fh:
      # slurm commands
      lines = (f"""\
#!/bin/bash 
#SBATCH --job-name="{xc_func}"
#SBATCH --account={slurm_config['account']}
#SBATCH --partition={slurm_config['partition']}
#SBATCH --ntasks={slurm_config['ntasks']}
#SBATCH --nodes={slurm_config['nodes']}
#SBATCH --cpus-per-task={slurm_config['cpus-per-task']}
#SBATCH --time={slurm_config['time']}
#SBATCH --mem={slurm_config['mem']}

{load_python_cmd}
{run_cmd} 
      """)

      lines = textwrap.dedent(lines)
      fh.writelines(lines)

    # slurm batch submit
    if DEBUG:
      cmd = debug_cmd
    else:
      cmd = f"sbatch {job_file}"
    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd=out_dir,
        stdout=None,
    )
    proc.wait()
