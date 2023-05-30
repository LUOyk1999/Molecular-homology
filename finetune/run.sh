unsup=$1
export CUDA_VISIBLE_DEVICES=$2

for dataset in clintox bace bbbp sider tox21 toxcast hiv muv
do
sh finetune_tune.sh $unsup $dataset
# nohup sh finetune_tune.sh $unsup $dataset > log_"$dataset_""$unsup" 2>&1 &
done