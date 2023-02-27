# Folder

* `pretrain/` contains codes for self-supervised pretraining.
* `finetune/` contains codes for finetuning on MoleculeNet benchmarks for evaluation.

## Requirements

python 3.7
```
pytorch                   1.8.1             
torch-geometric           1.7.0
rdkit                     2022.3.3
tqdm                      4.31.1
tensorboardx              1.6
```

All the necessary data files can be downloaded from the following links.
Download from [data](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip) (2.5GB), put it under `finetune/`, and unzip it.


## Generate PD

```
cd pretrain
python data_utils_NC.py
```

## Training
You can pretrain the model by
```
cd pretrain
python pretrain_supervised.py

# If pre-training is based on Hu's model:
python pretrain_supervised.py --input_model_file ./saved_model/masking
```

## Evaluation
You can evaluate the pretrained model by finetuning on downstream tasks
```
cd finetune
python finetune.py

or

cd finetune
sh run.sh
```
