# list=(0)
# list=(17 18 19 20 21)
# list=(2 3 4 5 6 9 10 11 12 13 16 17 18 19 22 23 24)
# rat=(0.2)
rat=(0.1 0.2)

for i in {1..24}
# for i in ${list[@]}
do
for j in ${rat[@]}
do 
echo ${i} ${j}
python exp/attribution_cifar.py --data-path /data8/donghun/cifar10/untracked --model-path /data8/donghun/cifar10/results/densenet/script_model.pt --method ${i} --device cuda:2 --type 2 --ratio ${j}
# python exp/attribution_mnist.py --data-path /data8/donghun/cifar10/untracked --device cuda:7 --type zero --ratio 0.2
done
done
