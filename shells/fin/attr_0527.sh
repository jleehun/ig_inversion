python exp/attribution/linear_interpolation.py --data-path /data8/donghun/cifar10/untracked --save-dir /home/dhlee/code/ig_inversion/results --model-path /data8/donghun/cifar10/results/densenet/script_model.pt --device cuda:6 --method image_gradient_ascent; python exp/attribution/linear_interpolation.py --data-path /data8/donghun/cifar10/untracked --save-dir /home/dhlee/code/ig_inversion/results --model-path /data8/donghun/cifar10/results/densenet/script_model.pt --device cuda:6 --method latent_gradient_descent; python exp/attribution/linear_interpolation.py --data-path /data8/donghun/cifar10/untracked --save-dir /home/dhlee/code/ig_inversion/results --model-path /data8/donghun/cifar10/results/densenet/script_model.pt --device cuda:6 --method latent_gradient_ascent;python exp/attribution/linear_interpolation.py --data-path /data8/donghun/cifar10/untracked --save-dir /home/dhlee/code/ig_inversion/results --model-path /data8/donghun/cifar10/results/densenet/script_model.pt --device cuda:7 --method image_simple_gradient_descent; python exp/attribution/linear_interpolation.py --data-path /data8/donghun/cifar10/untracked --save-dir /home/dhlee/code/ig_inversion/results --model-path /data8/donghun/cifar10/results/densenet/script_model.pt --device cuda:7 --method image_simple_gradient_ascent; python exp/attribution/linear_interpolation.py --data-path /data8/donghun/cifar10/untracked --save-dir /home/dhlee/code/ig_inversion/results --model-path /data8/donghun/cifar10/results/densenet/script_model.pt --device cuda:7 --method latent_simple_gradient_descent; python exp/attribution/linear_interpolation.py --data-path /data8/donghun/cifar10/untracked --save-dir /home/dhlee/code/ig_inversion/results --model-path /data8/donghun/cifar10/results/densenet/script_model.pt --device cuda:7 --method latent_simple_gradient_ascent; python exp/attribution/linear_interpolation.py --data-path /data8/donghun/cifar10/untracked --save-dir /home/dhlee/code/ig_inversion/results --model-path /data8/donghun/cifar10/results/densenet/script_model.pt --device cuda:6 --method image_fgsm; python exp/attribution/linear_interpolation.py --data-path /data8/donghun/cifar10/untracked --save-dir /home/dhlee/code/ig_inversion/results --model-path /data8/donghun/cifar10/results/densenet/script_model.pt --device cuda:6 --method image_simple_fgsm; python exp/attribution/linear_interpolation.py --data-path /data8/donghun/cifar10/untracked --save-dir /home/dhlee/code/ig_inversion/results --model-path /data8/donghun/cifar10/results/densenet/script_model.pt --device cuda:6 --method image_pgd; 


# interp_path  = {
#     # 'zero': '/home/dhlee/results/cifar10/image_linear_zero_interpolation.npy',
#     # 'expected': '/home/dhlee/results/cifar10/image_linear_expected_interpolation.npy',
    
#     # 'latent_linear': '/home/dhlee/results/cifar10/latent_linear_interpolation.npy',
    
#     'image_gradient_descent': '/home/dhlee/results/cifar10/image_gradient_descent_interpolation.npy', # descent
#     'image_gradient_ascent': '/home/dhlee/results/cifar10/image_gradient_ascent_interpolation.npy',            
#     'latent_gradient_descent': '/home/dhlee/results/cifar10/latent_gradient_descent_interpolation.npy',
#     'latent_gradient_ascent': '/home/dhlee/results/cifar10/latent_gradient_ascent_interpolation.npy',
    
#     'image_simple_gradient_descent': '/home/dhlee/results/cifar10/image_simple_gradient_descent_interpolation.npy', # descent
#     'image_simple_gradient_ascent': '/home/dhlee/results/cifar10/image_simple_gradient_ascent_interpolation.npy',            
#     'latent_simple_gradient_descent': '/home/dhlee/results/cifar10/latent_simple_gradient_descent_interpolation.npy',
#     'latent_simple_gradient_ascent': '/home/dhlee/results/cifar10/latent_simple_gradient_ascent_interpolation.npy',
    
#     'image_simple_fgsm': '/home/dhlee/results/cifar10/image_simple_fgsm_interpolation.npy',
#     'image_fgsm': '/home/dhlee/results/cifar10/image_fgsm_interpolation.npy',
#     'image_pgd': '/home/dhlee/results/cifar10/image_pgd_interpolation.npy',
# }[args.method]