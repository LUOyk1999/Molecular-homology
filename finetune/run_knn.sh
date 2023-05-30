
for dataset in bace bbbp sider clintox tox21 toxcast hiv muv
do
python knn.py --input_model_file graphcl_TDL --dataset $dataset
done