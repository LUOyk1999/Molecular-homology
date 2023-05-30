cd dataset
for dataset in clintox bace bbbp sider hiv muv tox21 toxcast zinc_standard_agent
do
cd $dataset
cd processed
rm -f geometric_data_processed.pt
cd ..
cd ..
done

# cd runs
# for dataset in clintox bace bbbp sider hiv muv tox21 toxcast
# do
# cd $dataset
# rm -f *.txt
# cd ..
# done
