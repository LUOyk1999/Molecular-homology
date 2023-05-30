#### GIN fine-tuning
unsup=$1
dataset=$2
split=scaffold

### for GIN
for runseed in 0 1 2 3 4 5 6 7 8 9
# for runseed in 0
do
model_file=${unsup}
python finetune_fix.py --input_model_file ../pretrain_PI/${model_file}.pth --split $split --runseed $runseed --gnn_type gin --dataset $dataset --epoch 100
done
