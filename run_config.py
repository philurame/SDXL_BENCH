import yaml
import subprocess
import os

ROOT = '/home/mdnikolaev/philurame'
PROJ_ROOT = os.path.join(ROOT, 'SDXL_METRICS')
with open(os.path.join(ROOT, 'wandb_key.txt'), 'r') as f:
  WANDB_KEY = f.read().strip()

MAX_SAMPLES = 10000

with open(os.path.join(PROJ_ROOT, 'config.yaml'), 'r') as file:
  config = yaml.safe_load(file)

generate = config['generate']

solvers = config['solvers']
schedulers = config['schedulers']
cachers_quantizers = config['cachers_quantizers']
nfes = config['nfes']
datasets = config['datasets']

strgenerate = 'generate' if generate else 'metrics'

from itertools import product
for dataset, nfe, solver, scheduler, cacher_quantizer in product(datasets, nfes, solvers, schedulers, cachers_quantizers):
  method =  f'{solver}_{scheduler}_{cacher_quantizer}_{nfe}'
  outdir = os.path.join(ROOT, f'_runs/{strgenerate}_{solver}')
  os.makedirs(outdir, exist_ok=True)

  # Build the script content
  script_content = f"""\
#!/bin/bash --login
#SBATCH --job-name={method}
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --constraint="[type_a|type_b|type_c|type_e]"
#SBATCH --output={outdir}/{method}-%j.log

module load Python/Anaconda_v03.2023

conda deactivate
conda activate philurame_venv

python3 {os.path.join(PROJ_ROOT, 'main.py')} \\
--root {PROJ_ROOT} \\
--key {WANDB_KEY} \\
--max_samples {MAX_SAMPLES} \\
--generate {generate} \\
--dataset {dataset} \\
--solver {solver} \\
--scheduler {scheduler} \\
--cacher_quantizer {cacher_quantizer} \\
--nfe {nfe}
"""

  # Submit the script content via sbatch
  command = ['sbatch']
  subprocess.run(command, input=script_content, text=True)