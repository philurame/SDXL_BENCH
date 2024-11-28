import yaml
import subprocess

with open('config.yaml', 'r') as file:
  config = yaml.safe_load(file)

generate    = config['generate']

solvers            = config['solvers']
schedulers         = config['schedulers']
cachers_quantizers = config['cachers_quantizers']
nfes               = config['nfes']
datasets           = config['datasets']

strgenerate = 'generate' if generate else 'metrics'


from itertools import product
for dataset, nfe, solver, scheduler, cacher_quantizer in product(datasets, nfes, solvers, schedulers, cachers_quantizers):
  method =  f'{solver}_{scheduler}'
  command = ['sbatch', 
            f'--output=_runs/{strgenerate}_{method}/script-%j.log', 
            'script.sbatch', 
            str(generate),
            dataset,
            solver, 
            scheduler,
            cacher_quantizer,
            str(nfe), 
            ]
  subprocess.run(command)