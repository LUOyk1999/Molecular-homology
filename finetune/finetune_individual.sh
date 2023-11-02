#### GIN fine-tuning
unsup=$1
split=scaffold

### for GIN
for task in 0 1
do
for runseed in 0 1 2 3 4 5 6 7 8 9
# for runseed in 0 1 2 3 4 5
do
model_file=${unsup}
python finetune_individual.py --input_model_file ../pretrain_PI/models/${model_file}.pth --split $split --runseed $runseed --gnn_type gin --dataset clintox --task $task --epoch 100
done
done