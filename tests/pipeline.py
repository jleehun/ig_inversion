import torch
import torch.nn.functional as F
import torchvision.transforms as T

from tqdm import tqdm

import numpy as np

from ig_pkg.models.generator import get_model
from ig_pkg.models.classifier import get_classifier
from ig_pkg.models.pretrained_models import get_pretrained_model

from ig_pkg.loss.focal_loss import FocalLoss
from ig_pkg.loss.metrics import ArcMarginProduct, AddMarginProduct

from ig_pkg.inputattribs.ig import ig
from ig_pkg.inputattribs.baseline_generator import get_baseline_generator

from ig_pkg.misc import process_heatmap, normalize_tensor, convert_to_img, label_to_class, pipeline_figure, na_imshow, tran

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def one_shot(gen_name, gen_path, gen_res,
             batch_size, 
             classifier, 
             classifier_name, 
             classifier_path,
             classifier_size, 
             class_path,
             to_device = True,
            ):
    if to_device:
        device = "cuda:0"
    else: device = "cpu"
    
    gen = get_model(gen_name, gen_path, gen_res).to(device)
    
    latent = torch.rand((batch_size, 512)).to(device)
    images = gen(latent).to(device)    
    
    if classifier:
        classifier = classifier.to(device)
    else:
        classifier = get_classifier(classifier_name, classifier_path)
        classifier = classifier.to(device) 
    
    transform = T.Compose([
                T.Resize(classifier_size, interpolation=T.functional.InterpolationMode.BILINEAR, antialias=True),
                T.CenterCrop(classifier_size),
                tran,
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])
    images = transform(images)
    
    class_names = np.load(class_path)
    
    score = classifier(images)#.to("cpu")        
    label = torch.argmax(score, dim = 1)
    predicted = label_to_class(label, class_names)
    
    res = {'gen_img': images,
           'pred': predicted,
    }
    return res

    
if __name__ =="__main__":
    temp_list = ['original']
    for mode_i in temp_list:
        mode = mode_i

        if mode == 15:
            num_labels=307
            classfier_path = '/root/pretrained/celebahq_whole_99.9736557006836.pth'
            original_img_path = '/root/data/CelebA_HQ_facial_identity_dataset/train'
            class_path = '/root/data/celebahq_identity/celebAHQ_identity_class_names.npy'
        elif mode == 'whole':
            num_labels=3819
            classfier_path ='/root/pretrained/celebahq_whole_99.9736557006836.pth'
            original_img_path = '/root/data/whole'
            class_path = '/root/data/celebahq_identity/identity_whole.npy'
        elif mode == 5:
            num_labels=2398
            classfier_path ='/root/pretrained/celebahq_train_99.90181732177734.pth'
            original_img_path = '/root/data/train'
            class_path = '/root/data/celebahq_identity/identity_5.npy'
        elif mode == 'original':
            num_labels=6217
            classfier_path ='/root/pretrained/celebahq_original_99.76000213623047.pth'
            original_img_path = '/root/data/identity_celebahq'
            class_path = '/root/data/celebahq_identity/identity_original.npy'

        clas = get_classifier('resnet', num_labels, classfier_path)

        for i in range(4):
            temp1 = one_shot(gen_name='stylegan', 
                     gen_path='/root/pretrained/stylegan-celebahq-1024x1024.pt', 
    #                 gen_path = '/root/pretrained/pggan_celebahq1024.pth',
                     gen_res=1024,
                     batch_size=5, 
                    classifier = clas,                 
                     classifier_name='resnet', 
                     classifier_path='nono',
                     classifier_size=224, 
                     class_path= class_path,
                     to_device = True,
                    )        

            pipeline_figure(img=temp1['gen_img'], predicted=temp1['pred'], image_path=original_img_path, fig_dir='/root/ig_inversion/figure/baseline/', fig_num=f'style_{mode}_{i}.png')
            print('good')
        torch.cuda.empty_cache()
    #     pipeline_figure(img=im, predicted=pr, image_path='/root/data/CelebA_HQ_facial_identity_dataset/train', fig_dir='/root/ig_inversion/figure/baseline/', fig_num=f'style_1_{i}.png')
