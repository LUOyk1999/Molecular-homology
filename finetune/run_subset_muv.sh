#### GIN fine-tuning
unsup=$1
split=scaffold

### for GIN
for threshold in 10000 20000 30000 40000
do
for runseed in 0 1 2 3 4 5 6 7 8 9
do
model_file=${unsup}
python finetune_subset.py --input_model_file ../pretrain_PI/models/${model_file}.pth --split $split --split_seed $runseed --gnn_type gin --dataset muv --epoch 100 --threshold $threshold 
done
done
