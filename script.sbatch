#!/bin/bash --login
#SBATCH --job-name=SDXLMETRICS        # Заместо aboba напишите ваше незвание эксперимента  
##SBATCH --error=_runs/{$SLURM_OUTP}/script-%j.err 
##SBATCH --output=_runs/{$SLURM_OUTP}/script-%j.log 
#SBATCH --gpus=1                      # Количество запрашиваемых гпу
#SBATCH --nodes=1                     # Количество запрашиваемых узлов
#SBATCH --cpus-per-task=8             # Выполнение расчёта на 8 ядрах CPU
#SBATCH --time=72:00:00               # Максимальное время выполнения (чч:мм:сс)
#SBATCH --constraint="[type_a|type_b|type_c|type_d]"   # Выбор, какие узлы использовать. 
																		 # Узлы a, b и c - содержат v100, d без гпу, e - содержит a100, простым смертным недоступен (нужно писать Айбеку)

echo $SLURM_OUTP

module load Python/Anaconda_v03.2023

conda deactivate
conda activate philurame_venv


# ./vqdm/quantization.sh
python3 SDXL_METRICS/main.py \
  --root /home/mdnikolaev/philurame/SDXL_METRICS \
  --key <YOUR WANDB KEY HERE> \
  --max_samples 10000 \
  --generate $1 \
  --dataset $2 \
  --solver $3 \
  --scheduler $4 \
  --cacher_quantizer $5 \
  --nfe $6