#### GIN fine-tuning

split=scaffold

### for GIN
for dataset in bace clintox bbbp sider
do
for threshold in 100 200 400 800 1000
do
for runseed in 0 1 2 3 4 5 6 7 8 9
do
# python finetune_subset.py --input_model_file ../pretrain_PI/models/${model_file}.pth --split $split --runseed $runseed --gnn_type gin --dataset bace --epoch 100 --threshold $threshold 
python PI_SSVM_subset.py --threshold $threshold --dataset $dataset --split_seed $runseed
done
done
done
