import textwrap
import subprocess
from pathlib import Path
import shutil

DEBUG = False

# out directory
out_dir = Path('ea_out')
out_dir.mkdir(parents=True, exist_ok=True)

# copy over files
status = shutil.copy2('ea_atoms.yaml', out_dir)

run_file = '../ea_runs.py'
job_file = f'ea.job'
run_cmd = f'srun python {run_file}'
debug_cmd = f'python3 {run_file}'
with open(out_dir / job_file, "w") as fh:
  # slurm commands
  lines = (f"""\
      #!/bin/bash
      #SBATCH --job-name=ea
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
