list=(0 7 14 20 1 8 15 21)
rat=(0.1 0.2)

for i in ${list[@]}
do
for j in ${rat[@]}
do 
echo ${i} ${j}
python exp/attribution.py --data-path /data8/donghun/cifar10/untracked --method ${i}--model-path /data8/donghun/cifar10/results/densenet/script_model.pt --device cuda:6 --ratio ${j}
done
done
