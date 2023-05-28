# list="2 4 6 8 10"
# for i in ${list}

# list= (linear_latent_linear linear_image_gradient_descent linear_image_gradient_ascent linear_latent_gradient_descent linear_latent_gradient_ascent linear_image_simple_gradient_descent linear_image_simple_gradient_ascent linear_latent_simple_gradient_descent linear_latent_simple_gradient_ascent linear_image_simple_fgsm linear_image_fgsm linear_image_pgd linear_image_cw)
list=('image_cw' 'linear_image_simple_gradient_descent' 'linear_image_simple_gradient_ascent' 'linear_latent_simple_gradient_descent' 'linear_latent_simple_gradient_ascent'  'linear_image_simple_fgsm' 'linear_image_fgsm' 'linear_image_pgd' 'linear_image_cw'
)
rat=(0.15 0.1 0.05)

for i in ${list[@]}
do
for j in ${rat[@]}
do 
echo ${i} ${j}
python exp/attribution_temp.py --data-path /data8/donghun/cifar10/untracked --model-path /data8/donghun/cifar10/results/densenet/script_model.pt --device cuda:3 --method ${i} --ratio ${j};
done
done


# "linear_latent_linear, linear_image_gradient_descent, linear_image_gradient_ascent, linear_latent_gradient_descent, linear_latent_gradient_ascent, linear_image_simple_gradient_descent, linear_image_simple_gradient_ascent, linear_latent_simple_gradient_descent, linear_latent_simple_gradient_ascent, linear_image_simple_fgsm, linear_image_fgsm, linear_image_pgd, image_cw, linear_image_cw"