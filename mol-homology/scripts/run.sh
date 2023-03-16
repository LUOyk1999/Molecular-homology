#!/bin/bash
set -e

REPO=$PWD

source activate polymers
export CUDA_VISIBLE_DEVICES=0
#export PYTHONPATH=$PYTHONPATH:$REPO:../polymers/

MDIR="/u/vat/Molecular-homology/mol-homology/pretrain_PI/pretraining_model/"
MODEL="PI_predictor_atom.pth"
DATA="sider"

echo Starting run!
echo Config: $MODEL $DATA


cd finetune
python finetune.py --input_model_file "${MDIR}/${MODEL}" --dataset $DATA

