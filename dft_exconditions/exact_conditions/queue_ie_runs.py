import textwrap
import subprocess
from pathlib import Path
import shutil

DEBUG = False

xcs = [
    'mgga_x_revm06_l,mgga_c_revm06_l',
    'HYB_MGGA_X_M06_SX,MGGA_C_M06_SX',
    'hyb_mgga_x_m11,mgga_c_m11',
    'hyb_mgga_x_m05,mgga_c_m05',
    'hyb_mgga_x_revm06,mgga_c_revm06',
    'mgga_x_m06_l,mgga_c_m06_l',
]
completed_xcs = [
    'hyb_mgga_x_m08_hx,mgga_c_m08_hx',
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
out_dir = Path('ie_out')
out_dir.mkdir(parents=True, exist_ok=True)

# copy over files
status = shutil.copy2('ie_atoms.yaml', out_dir)

for xc in xcs:

  run_file = '../ie_runs.py'
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
