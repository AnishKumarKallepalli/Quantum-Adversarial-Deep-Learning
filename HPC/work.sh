#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 3-0:00 # time (D-HH:MM)
#SBATCH --job-name="work.sh"
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err

spack load anaconda3@2021.05%gcc@8.5.0

conda init bash
eval "$(conda shell.bash hook)"
conda activate /home/aneesh/.conda/envs/mypython3

python  /home/aneesh/Anish/quantum_deep_learning/run.py

