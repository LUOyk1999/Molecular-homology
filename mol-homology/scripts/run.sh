#!/bin/bash
set -e
#jbsub -queue x86_24h -mem 100g -cores 10+1 -require a100 -out oc1 -err c1 scripts/run.sh

REPO=$PWD

source activate polymers
export CUDA_VISIBLE_DEVICES=0

MDIR="/u/vat/Molecular-homology/mol-homology/pretrain_PI/pretraining_model/"
MODEL="PI_predictor_atom.pth"
DATA="sider"

echo Starting run!
echo Config: $MODEL $DATA


cd finetune
python finetune.py --input_model_file "${MDIR}/${MODEL}" --dataset $DATA --dir_data "/dccstor/vthost-data/hdata/dataset"

