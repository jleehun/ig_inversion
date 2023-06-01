
rat=(19 24)

for j in ${rat[@]}
do 
echo ${j}
python exp/attribution/baseline_gen.py --data-path /root/data/cifar10 --model-path /root/data/cifar10/cifar10/results/densenet/script_model.pt --device cuda:0 --type ${j}
done 
