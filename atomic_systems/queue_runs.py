import textwrap
import subprocess
from pathlib import Path
import shutil

import pylibxc

DEBUG = False

xcs = [
    'm06',
    'pbe',
    'sogga11',
    'scan',
    'r2scan',
    'b3lyp',
    'mn15',
    'HYB_GGA_XC_B97',
    'gga_x_am05,gga_c_am05',
    #'HYB_GGA_XC_CASE21',
]

# out directory
out_dir = Path('out')
out_dir.mkdir(parents=True, exist_ok=True)

# copy over files
status = shutil.copy2('ie_atoms.yaml', out_dir)

for xc in xcs:

  run_file = '../runs.py'
  job_file = f'{xc}.job'
  run_cmd = f'srun python {run_file} {xc} '
  debug_cmd = f'python3 {run_file} {xc} '
  with open(out_dir / job_file, "w") as fh:
    # slurm commands
    lines = (f"""\
        #!/bin/bash
        #SBATCH --job-name="{xc}"
        #SBATCH --account=burke
        #SBATCH --partition=sib2.9,nes2.8
        #SBATCH --ntasks=1
        #SBATCH --nodes=1
        #SBATCH --cpus-per-task=8
        #SBATCH --time=10:00:00
        #SBATCH --mem=32G

        ml purge
        ml miniconda/3/own
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
