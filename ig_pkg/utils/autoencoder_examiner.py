
import numpy as np 
import torch 
import matplotlib.pyplot as plt 


def denormalize(tensor, means, stds):
    denormalized = tensor.clone()
    for channel, mean, std in zip(denormalized, means, stds):
        channel.mul_(std).add_(mean)
    return denormalized



class AutoencoderExaminer:
    
    @staticmethod
    def reconstruct_input(autoencoder, x, **kwargs):
        training = autoencoder.training
        autoencoder.eval()
        with torch.no_grad():        
            x_hat, loss_dict, info = autoencoder(x, **kwargs)
        if training:
            autoencoder.train()
        return x_hat, loss_dict, info  # multiple outputs  
            
    @staticmethod
    def reconstruct_latent(autoencoder, z, **kwargs):
        training = autoencoder.training
        autoencoder.eval()
        with torch.no_grad():        
            x_hat = autoencoder.decode(z, **kwargs)
        if training:
            autoencoder.train()
        return x_hat  # multiple outputs  

            
    @staticmethod
    def get_classifier_latent_direction(autoencoder, classifier, z, y):
        
        # clean the gradients 
        autoencoder.zero_grad()
        classifier.zero_grad()        
        
        z = torch.autograd.Variable(z, requires_grad=True).to(z.dtype)
        x_hat = autoencoder.decode(z)
        output = classifier(x_hat)
        score = torch.softmax(output, dim=-1)
        class_score = torch.FloatTensor(z.size(0), output.size()[-1]).zero_().to(z.device).type(z.dtype)
        class_score[:,y] = score[:,y]
        output.backward(gradient=class_score)
        return z.grad.detach()
            
    @staticmethod
    def save_original_and_reconstruction(x_hat, x, means, stds, save_dir=None, rows=False, **kwargs ):
        n = x_hat.size(0)
        size = tuple(x_hat.size()[1:])
        if rows:
            fig, axes = plt.subplots(n, 2, **kwargs)    
            for i in range(n):
                origin = denormalize(x[i].cpu().detach(), means, stds)
                recons = denormalize(x_hat[i].cpu().detach(), means, stds)
                if origin.size(0) < origin.size(1):
                    origin = origin.permute(1,2,0)
                    recons = recons.permute(1,2,0)
                origin = (origin.numpy()*255).clip(0,255).astype(np.int)
                recons = (recons.numpy()*255).clip(0,255).astype(np.int)
                print(origin.max(), origin.min(), recons.max(), recons.min(), )
                axes[i,0].imshow(origin)
                axes[i,1].imshow(recons)
                axes[i,0].set_yticks([])
                axes[i,0].set_xticks([])
                axes[i,1].set_yticks([])
                axes[i,1].set_xticks([])
            axes[0,0].set_title(f"Original \n{size}")
            axes[0,1].set_title(f"Reconstructed\n{size}")
        else:
            fig, axes = plt.subplots(2, n, **kwargs)    
            for i in range(n):
                origin = denormalize(x[i].cpu().detach(), means, stds)
                recons = denormalize(x_hat[i].cpu().detach(), means, stds)
                if origin.size(0) < origin.size(1):
                    origin = origin.permute(1,2,0)
                    recons = recons.permute(1,2,0)
                origin = (origin.numpy()*255).clip(0,255).astype(np.int)
                recons = (recons.numpy()*255).clip(0,255).astype(np.int)
                
                axes[0,i].imshow(origin)
                axes[1,i].imshow(recons)
                axes[0,i].set_yticks([])
                axes[0,i].set_xticks([])
                axes[1,i].set_yticks([])
                axes[1,i].set_xticks([])
            axes[0,0].set_ylabel(f"Original \n{size}")
            axes[1,0].set_ylabel(f"Reconstructed\n{size}")
            
        plt.tight_layout()
        plt.show()
        if save_dir is not None:
            import os 
            if not os.path.exists(os.path.dirname(save_dir)):
                os.makedirs(os.path.dirname(save_dir))
            plt.savefig(save_dir)
            
    @staticmethod
    def save_reconstruction(x_hat, means, stds, save_dir=None, rows=False, **kwargs ):
        n = x_hat.size(0)
        size = tuple(x_hat.size()[1:])
        if rows:
            fig, axes = plt.subplots(n, 1, **kwargs)    
            for i in range(n):
                recons = denormalize(x_hat[i].cpu().detach(), means, stds)
                if recons.size(0) < recons.size(1):
                    recons = recons.permute(1,2,0)
                recons = (recons.numpy()*255).clip(0,255).astype(np.int)

                axes[i].imshow(recons)
                axes[i].set_yticks([])
                axes[i].set_xticks([])
            axes[0].set_title(f"Reconstructed\n{size}")
        else:
            fig, axes = plt.subplots(1, n, **kwargs)    
            for i in range(n):
                recons = denormalize(x_hat[i].cpu().detach(), means, stds)
                if recons.size(0) < recons.size(1):
                    recons = recons.permute(1,2,0)
                recons = (recons.numpy()*255).clip(0,255).astype(np.int)
                axes[i].imshow(recons)
                axes[i].set_yticks([])
                axes[i].set_xticks([])
            axes[0].set_ylabel(f"Reconstructed\n{size}")
            
        plt.tight_layout()
        plt.show()
        if save_dir is not None:
            import os 
            if not os.path.exists(os.path.dirname(save_dir)):
                os.makedirs(os.path.dirname(save_dir))
            plt.savefig(save_dir)            

if __name__ == "__main__":
    class SIMPLEClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(32*32, 10)
        def forward(self, x):
            x = x.flatten(start_dim=1)
            return self.fc(x)        
        
    from bm.vision.models.autoencoder.modules.simple import EncoderModule, DecoderModule
    from bm.vision.models.autoencoder.modules.latents import IdentityBottleNeck, NormalBottleNeck
    from bm.vision.models.autoencoder.ae import AutoEncoder
    from bm.vision.models.autoencoder.vae import VAE
    
    in_size = (1,32,32)
    encoder = EncoderModule(1, [64, 128, 256])
    decoder = DecoderModule(1, [64, 128, 256])
    bottleneck_ae = IdentityBottleNeck()
    bottleneck_vae = NormalBottleNeck((256, 4,4), 64, 32)
    ae = AutoEncoder(encoder, decoder, bottleneck_ae, in_size)
    vae = VAE(encoder, decoder, bottleneck_vae, in_size, 0.005)
    classifier = SIMPLEClassifier()
    
    input= torch.randn(2, *in_size)
    
    for model in [ae, vae]:
        x_hat = AutoencoderExaminer.reconstruct_input(model, input)[0]
        print(x_hat.size())
        AutoencoderExaminer.save_original_and_reconstruction(x_hat, input, [0], [0.5], f'results/{model.__class__.__name__}_recon.pdf' )
        z = model.sample(num_samples=1)
        grad = AutoencoderExaminer.get_classifier_latent_direction(model, classifier, z, y=0)
        z -= grad  * 1.5
        print(z.size())