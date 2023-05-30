# list=(2 3 4 5 6 9 10 11 12 13 16 17 18 19 22 23 24)
rat=(0.01 0.1 0.2)

for i in {0..24}
do
for j in ${rat[@]}
do 
echo ${i} ${j}
python exp/attribution_mnist.py --data-path /data8/donghun/cifar10/untracked --device cuda:5 --type ${i} --ratio ${j}
done
done
