#!/bin/bash
set -e

cd /dccstor/vthost-data/hdata

wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
unzip chem_dataset.zip
wget -O PI.zip "https://drive.google.com/file/d/1rzg5PdyhlQ17_lSqgy524tqJ6xl-CB8r&export=download&confirm=yes"
unzip PI.zip
