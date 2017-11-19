#!/bin/sh
#SBATCH --job-name=nti_train
#SBATCH --output=slurm_%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=24000
#SBATCH -t24:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user=tr1312@nyu.edu  # email me when the job ends


module load h5py/intel/2.7.0rc2
module load pytorch/0.2.0_1
module load numpy/intel/1.13.1
module load pandas/intel/py2.7/0.20.3
module load scikit-learn/intel/0.18.1

source /home/tr1312/pyenv/bin/activate
#module load chainer/1.12.0

python "/scratch/tr1312/nti/train_nti.py"  \
--gpu 1 \
--snli "/scratch/tr1312/data/snli_1.0" \
--glove "/scratch/tr1312/data/glove.6B/glove.6B.300d.txt"

