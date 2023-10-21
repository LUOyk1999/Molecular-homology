## Installation
Tested with Python 3.7, PyTorch 1.12.1, and PyTorch Geometric 2.2.0.
The dependencies are managed by [conda]:

```
pip install -r requirements.txt
```


## Requirements

* `pretrain_PI/` contains codes for self-supervised pretraining.
* `finetune/` contains codes for finetuning on MoleculeNet benchmarks for evaluation.

All the necessary data files can be downloaded from the following links.
Download from [chem public resources](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip) (provided by Weihua Hu, Bowen Liu, and others. Strategies for pre-training graph neural networks. ICLR, 2020), put it under `./finetune/`, and unzip it. Remember to delete the old `geometric_data_processed.pt` file.
You can use this sh script to delete these files directly:
```
cd finetune
sh remove_old.sh
```

# Improving Self-supervised Molecular Representation Learning using Persistent Homology


## Calculate Topological Fingerprints (PI) and Training

```
cd pretrain_PI
sh PI.sh
```

Then you can pretrain TAE by
```
python TAE.py
```

you can pretrain TAE+contextpred (TAE+edgepred, TAE+masking) by
```
python pretrain_contextpred.py

python pretrain_edgepred.py

python pretrain_masking.py
```

You can pretrain GraphCL+TDL by
```
python pretrain_graphcl_TDL.py
```

You can pretrain GraphLoG+TDL by
```
python pretrain_graphlog_TDL.py
```

You can pretrain JOAO+TDL by
```
python JOAO_TDL.py
```

You can pretrain SimGRACE+TDL by
```
python pretrain_simgrace_TDL.py
```

Calculate the Pearson correlation coefficient between real PIs and reconstructed PIs
```
python model_PI_pearson.py
```

Similarity histograms of PIs 
```
python Pi_similarity_h.py
```

The pre-trained models `.pth` we provide.
```
graphcl_TDL.pth
graphlog_TDL.pth
joao_TDL.
simgrace_TDL.pth
tae.pth
tae_contextpred.pth
```

## Evaluation

Primary Results: you can evaluate the pretrained model by finetuning on downstream tasks
```
cd finetune
python finetune.py --input_model_file ../pretrain_PI/graphcl_TDL.pth --dataset bace

or

sh run.sh graphcl_TDL 0
sh run.sh tae 0
sh run.sh tae_contextpred 0
```

Linear probing: molecular property prediction

```
sh run_fix.sh graphcl_TDL 0
```

Linear probing: Pearson correlation coefficients between distances in embedding space and distances between corresponding PIs
```
sh run_PI.sh graphcl_TDL 0
```

5-nearest neighbors classifier
```
sh run_knn.sh
```

Performance over smaller datasets
```
sh run_subset_bace.sh graphcl_TDL
sh run_subset_bbbp.sh graphcl_TDL
sh run_subset_clintox.sh graphcl_TDL
sh run_subset_hiv.sh graphcl_TDL
sh run_subset_muv.sh graphcl_TDL
sh run_subset_sider.sh graphcl_TDL
sh run_subset_tox21.sh graphcl_TDL
sh run_subset_toxcast.sh graphcl_TDL
```

Individual Tasks
```
sh finetune_individual.sh graphlog_TDL
```

Ablation Studies: TAE (Concatenation during FT)
```
sh run_concat.sh
```

Results of non pre-training method
```
python PI_SSVM.py
python PI_XGB.py
```