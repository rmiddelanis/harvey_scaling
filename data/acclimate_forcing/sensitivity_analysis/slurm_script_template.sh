#!/bin/bash
#SBATCH --qos=+++qos+++
#SBATCH --partition=+++partition+++
#SBATCH --job-name=sensitivity
#SBATCH --account=acclimat
#SBATCH --output=test-%j.out
#SBATCH --error=test-%j.err
#SBATCH --workdir='+++workdir+++'
#SBATCH --cpus-per-task=+++num_cpu+++
#SBATCH --export=ALL,OMP_PROC_BIND=FALSE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
/home/robinmid/repos/old_acclimate/acclimate/build/acclimate +++workdir+++/settings.yml
