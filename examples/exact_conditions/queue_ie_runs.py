import textwrap
import subprocess
from pathlib import Path
import shutil

import pylibxc
import yaml

DEBUG = False

xcs = [
    'hyb_mgga_x_m08_hx,mgga_c_m08_hx',
    'm06',
    'pbe',
    'sogga11',
    'scan',
    'b3lyp',
    'HYB_GGA_XC_B97',
    'gga_x_am05,gga_c_am05',
    #'HYB_GGA_XC_CASE21',
]

# out directory
out_dir = Path('ie_out')
out_dir.mkdir(parents=True, exist_ok=True)

# copy over files
status = shutil.copy2('ie_atoms.yaml', out_dir)

# slurm config
with open('slurm_config.yaml', 'r') as file:
  slurm_config = yaml.safe_load(file)

  load_python_cmd = slurm_config['load_python_cmd']
  # separate commands by line breaks
  if isinstance(load_python_cmd, list):
    load_python_cmd = '\n'.join(load_python_cmd)

for xc in xcs:

  run_file = '../ie_runs.py'
  job_file = f'{xc}.job'
  run_cmd = f'srun python {run_file} --xc {xc} '
  debug_cmd = f'python3 {run_file} --xc {xc} '
  with open(out_dir / job_file, "w") as fh:
    # slurm commands
    lines = (f"""\
#!/bin/bash 
#SBATCH --job-name="{xc}"
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
