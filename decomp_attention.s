#!/bin/sh
#SBATCH --job-name=decomp_gpu
#SBATCH --output=slurm_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=24GB
#SBATCH --mail-type=END
#SBATCH --mail-user=da1933@nyu.edu
#SBATCH --gres=gpu:1

module load h5py/intel/2.7.0rc2
module load pytorch/0.2.0_1
module load numpy/intel/1.13.1

cd /scratch/da1933/da

python DecomposableAttentionGPU.py


