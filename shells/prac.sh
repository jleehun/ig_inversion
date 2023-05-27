# list="2 4 6 8 10"
# for i in ${list}

list="image_cw"
rat="0.2, 0.1"

for i in ${list}
do
for j in ${rat}
do 
echo ${i} ${j}
python exp/attribution.py --data-path /data8/donghun/cifar10/untracked --model-path /data8/donghun/cifar10/results/densenet/script_model.pt --device cuda:2 --method ${i} --ratio ${j};
done
done