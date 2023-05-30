
rat=(14 15 16 17 18 19)

for j in ${rat[@]}
do 
echo ${j}
python exp/attribution/baseline_gen.py --data-path /data8/donghun/cifar10/untracked --model-path /data8/donghun/cifar10/results/densenet/script_model.pt --device cuda:4 --type ${j}
done 
