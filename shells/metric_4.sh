# list=(2 3 4 5 6 9 10 11 12 13 16 17 18 19 22 23 24)
# list=(0 7 14 20 1 8 15 21)

list=(one half minus1 minus5)
# list=('one' 'half' 'minus1' 'minus5')
rat=(0.1 0.2)

# 0.01 0.05 0.15 all
# 0.1 (2 3 4 5 6 9 10 11 12 13 16 17 18 19 22 23 24)
for i in ${list[@]}
do
for j in ${rat[@]}
do 
echo ${i} ${j}
python exp/attribution_cifar.py --data-path /data8/donghun/cifar10/untracked --method ${i} --model-path /data8/donghun/cifar10/results/densenet/script_model.pt --device cuda:7 --ratio ${j}
done
done
