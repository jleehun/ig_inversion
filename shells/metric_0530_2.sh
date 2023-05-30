
rat=(7 8 9 10 11 12 13)

for j in ${rat[@]}
do 
echo ${j}
python exp/attribution/baseline_gen.py --data-path /data8/donghun/cifar10/untracked --model-path /data8/donghun/cifar10/results/densenet/script_model.pt --device cuda:3 --type ${j}
done 
