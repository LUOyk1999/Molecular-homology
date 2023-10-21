#### GIN fine-tuning
unsup=$1
split=scaffold

### for GIN
for threshold in 3000 6000 9000 20000 30000
do
for runseed in 0 1 2 3 4 5 6 7 8 9
# for runseed in 0
do
model_file=${unsup}
python finetune_subset.py --input_model_file ../pretrain_PI/${model_file}.pth --split $split --split_seed $runseed --gnn_type gin --dataset hiv --epoch 100 --threshold $threshold 
done
done