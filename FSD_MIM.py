"""Implementation of sample attack."""
import os
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
from attack_methods import DI,gkern
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image
from dct import *
from loader import ImageNet
from torch.utils.data import DataLoader
import argparse
import pretrainedmodels

from dct import *
from Normalize import Normalize, TfNormalize
from torch import nn
from torch_nets import (
    tf_inception_v3,
    tf_inception_v4,
    tf_resnet_v2_50,
    tf_resnet_v2_101,
    tf_resnet_v2_152,
    tf_inc_res_v2,
    tf_adv_inception_v3,
    tf_ens3_adv_inc_v3,
    tf_ens4_adv_inc_v3,
    tf_ens_adv_inc_res_v2,
    )
# num_workers=0


parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='./dataset/images.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='./dataset/images', help='Input directory with images.')
parser.add_argument('--model_dir', type=str, default='./models/', help='model directory.') 
parser.add_argument('--model_name', type=str, default='tf_inception_v3', help='source model name.') # 
parser.add_argument('--output_dir', type=str, default='./outputs/xxx/', help='Output directory with adversarial images.') #
parser.add_argument("--batch_size", type=int, default=30, help="How many images process at one time.") #
parser.add_argument("--N", type=int, default=5, help="The copy number ") # 
parser.add_argument('--bound', type=float, default= 45 , help='random noise bound')
parser.add_argument('--line', type=int, default= 280 , help='length parameter')
parser.add_argument('--mean', type=float, default=np.array([0.5, 0.5, 0.5]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.5, 0.5, 0.5]), help='std.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.") # 
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

transforms = T.Compose(
    [T.Resize(299), T.ToTensor()]
)

def normalize(grad,opt=2):
    if opt==0:
        nor_grad=grad
    elif opt==1: # 
        abs_sum=np.sum(np.abs(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/abs_sum
    elif opt==2:
        square = np.sum(np.square(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/np.sqrt(square) # 
    return nor_grad

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def save_image(images,names,output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))  
        img.save(output_dir + name)

T_kernel = gkern(7, 3) # 3,1,7,7

def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'tf_inception_v3':
        net = tf_inception_v3
    elif net_name == 'tf_inception_v4':
        net = tf_inception_v4
    elif net_name == 'tf_resnet_v2_50':
        net = tf_resnet_v2_50
    elif net_name == 'tf_resnet_v2_101':
        net = tf_resnet_v2_101
    elif net_name == 'tf_resnet_v2_152':
        net = tf_resnet_v2_152
    elif net_name == 'tf_inc_res_v2':
        net = tf_inc_res_v2
    elif net_name == 'tf_adv_inception_v3':
        net = tf_adv_inception_v3
    elif net_name == 'tf_ens3_adv_inc_v3':
        net = tf_ens3_adv_inc_v3
    elif net_name == 'tf_ens4_adv_inc_v3':
        net = tf_ens4_adv_inc_v3
    elif net_name == 'tf_ens_adv_inc_res_v2':
        net = tf_ens_adv_inc_res_v2
    else:
        print('Wrong model name!')

    model = nn.Sequential(
        
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        TfNormalize('tensorflow'),
        
        net.KitModel(model_path).eval().to(device))  ##  net.KitModel(model_path).eval().cuda(),)
    
    return model



### Details will be completed as soon as the paper is accepted ###
def FSD_MIM(images, gt, model, min, max):
    image_width = opt.image_width
    momentum = opt.momentum
    num_iter = 10
    eps = opt.max_epsilon / 255.0
    
    bound = opt.bound / 255.0
    line = opt.line
    
    alpha = eps / num_iter
    x = images.clone() # clone
    grad = 0
    N = opt.N 

    for i in range(num_iter):
        
            
        ne_allgrad = 0 # 
        for n in range(N):    # 
        
            randnoise = torch.randn_like(x).uniform_(-bound, bound)
            randnoise = randnoise.to(device)
            x_n = (x + randnoise).to(device)
            x_n_dct = dct_2d(x_n)     
            x_n_dct[:,:, :line, :line] = x_n_dct[:,:, :line, :line] * (1 - n / N)
            x_n_dct[:,:, line:, line:] = x_n_dct[:,:, line:, line:] * (1 + n / N)            
            rd = clip_by_tensor(torch.normal(torch.tensor(1.0),torch.tensor(1.0)), 1 - n / N, 1 + n / N)
            x_n_dct[:,:, line:, :line] = x_n_dct[:,:, line:, :line] * rd
            x_n_dct[:,:, :line, line:] = x_n_dct[:,:, :line, line:] * rd
            x_neighbor = idct_2d(x_n_dct)
            x_neighbor = V(x_neighbor, requires_grad = True)
            ne_output = model(x_neighbor)                 
            
            ## batchsize == 1 
            # ne_loss = F.cross_entropy(ne_output[0].unsqueeze(0), gt)
            
            
            ## batchsize not 1 
            ne_loss = F.cross_entropy(ne_output[0], gt)
            
            
            ne_loss.backward() 
            ne_allgrad += x_neighbor.grad.data

        noise = ne_allgrad / N

        
        # ### grad drop (effective for adversarially trained models) ###
        # noise_dct = dct_2d(noise)
                
        # noise_dct_sum = torch.sum(torch.abs(noise_dct),[1])
        # noise_dct_reshape = noise_dct_sum.reshape(x.shape[0], -1)
        # top_k = torch.topk(noise_dct_reshape, opt.k, dim=1)[0][:,-1]
        # top_k = top_k.unsqueeze(1)
        # top_k = top_k.unsqueeze(1)
        # m_small = (noise_dct_sum < top_k).float()
        # m_large = 1 - m_small
        # m1 = opt.gamma * m_small + m_large
        # m = torch.stack((m1,m1,m1), dim=1)
        # noise_dct_drop = m * noise_dct
        
        # noise = idct_2d(noise_dct_drop)      
        # ### over ###

        
        
        # TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
        # noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

        ## MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise

        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()
##################################################################




def main():

    ## model = torch.nn.Sequential(Normalize(opt.mean, opt.std),
    ##                             pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda())
    # model = torch.nn.Sequential(Normalize(opt.mean, opt.std),
    #                             pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().to(device))

    model = get_model(opt.model_name, opt.model_dir) #  [-1,1]


    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=0)  #  

    for images, images_ID,  gt_cpu in tqdm(data_loader):

        ## gt = gt_cpu.cuda()
        ## images = images.cuda()
        gt = gt_cpu.to(device)
        images = images.to(device)              
        
        images_min = clip_by_tensor(images - opt.max_epsilon / 255.0, 0.0, 1.0)
        images_max = clip_by_tensor(images + opt.max_epsilon / 255.0, 0.0, 1.0)

        adv_img = FSD_MIM(images, gt, model, images_min, images_max)
        adv_img_np = adv_img.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_image(adv_img_np, images_ID, opt.output_dir)

if __name__ == '__main__':
    main()
