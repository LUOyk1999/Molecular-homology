export CUDA_VISIBLE_DEVICES=0

for dataset in clintox bace bbbp sider tox21 toxcast hiv muv
do
sh finetune_concat.sh $dataset
done