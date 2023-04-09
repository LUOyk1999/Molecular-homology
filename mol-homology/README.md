## Requirements
python 3.7
```
pytorch                   1.12.1            
torch-geometric           2.2.0
rdkit                     2022.3.3
tqdm                      4.64.0
```

* `pretrain_PI/` contains codes for our pre-training.
* `finetune/` contains codes for fine-tuning on MoleculeNet benchmarks for evaluation.

1. All the necessary data files can be downloaded from the following links.
   Download from [data](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip) (2.5GB), put it under `./finetune/`, and unzip it. Remember to delete the old `geometric_data_processed.pt` file.
   You can use this sh script to delete these files directly:

```
cd finetune
sh remove_old.sh
```
2. Download from [PI](https://drive.google.com/file/d/1dIpehI1CTICguRjm-4o_RjQUvZrqfQwC/view?usp=share_link), put it under `./pretrain_PI/`, and unzip it.

# Self-supervised representation learning over molecules using homology

## Training
```
cd pretrain_PI
```
You can pretrain PI predictor by
```
python PIP.py
```
## Evaluation

You can evaluate the pretrained model by finetuning on downstream tasks
```
cd finetune
python finetune.py --input_model_file ../pretrain_PI/pretrain_models/PIP_atom_hks_200.pth --dataset bace

or

cd finetune
sh run.sh PIP_atom_hks_200 0
```
