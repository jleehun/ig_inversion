# ${list[@]}
list=(17 18 19 20 21)
rat=(0.2)

# 0.01 0.05 0.15 all
# 0.1 (2 3 4 5 6 9 10 11 12 13 16 17 18 19 22 23 24)
for i in ${list[@]}
do
echo ${i} 
python exp/attribution/baseline_gen_mnist.py --data-path /data8/donghun --device cuda:7 --type ${i}
done
