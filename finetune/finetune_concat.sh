#### GIN fine-tuning
dataset=$1
split=scaffold

### for GIN
for runseed in 0 1 2 3 4 5 6 7 8 9
do
model_file=${unsup}
python finetune_concat.py --input_model_file concat --split $split --runseed $runseed --gnn_type gin --dataset $dataset
done
