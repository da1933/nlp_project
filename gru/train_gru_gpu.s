#!/bin/sh
#SBATCH --job-name=12
#SBATCH --output=slurm_%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH -t168:00:00
#SBATCH --mail-type=END  # email me when the job ends
#SBATCH --mail-user=tr1312@nyu.edu

module load h5py/intel/2.7.0rc2
module load pytorch/0.2.0_1
module load numpy/intel/1.13.1
module load pandas/intel/py2.7/0.20.3


python train_baseline_snli.py \
--train_file "/scratch/tr1312/decomp_attn/data/snli_preprocess/train.hdf5" \
--dev_file "/scratch/tr1312/decomp_attn/data/snli_preprocess/val.hdf5" \
--test_file "/scratch/tr1312/decomp_attn/data/snli_preprocess/test.hdf5" \
--w2v_file "/scratch/tr1312/decomp_attn/data/snli_preprocess/glove.hdf5" \
--log_dir "/scratch/tr1312/gru/log/" \
--log_fname "ada03.log" \
--model_path "/scratch/tr1312/gru/output/ada03/" \
--gpu_id 0 \
--lr 0.0032342 \
--dropout 0.136797 \
--optimizer Adagrad \
--embedding_size 300 \
--hidden_size 150 \
--max_length -1 \
--display_interval 1000 \
--epoch 150 \
--weight_decay 0.000082