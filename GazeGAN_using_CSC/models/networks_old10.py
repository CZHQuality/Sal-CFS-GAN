# We add the saliency losses in this script, which are CC, NSS, KL and L1, and we use L1 loss to replace VGG loss (represented by Class VGGLoss)
# We add the resblock as bottle-neck layer of UNet
# This code is for inference the intermediate representation (feature map) of the pre-trained saliency model
# This code is for constructing the SalGAN network architecture using this whole framework
### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
# we perform the real SALICON in this code, which contains two VGG streams with different resolutions
# CUDA_VISIBLE_DEVICES=1 python3 train.py --name label2city_512p --no_instance --label_nc 0 --no_ganFeat_loss --netG global_unet --resize_or_crop none --continue_train
# This code contains inception-resnet-deformableconv module
# This code contains basic module: "inception-resnet-deformableconv module + SEmodule", and contains only one basic module
# We extend this code to include more than one (in fact, several) basic modules in the backbone network (basic modules: "inception-resnet-deformableconv module + SEmodule"), and without VGGloss, 
# to do: 1. multiply the saliency map with the input original image, to establish a "conditional perceptually loss": use VGG16 to calculate the difference between two products from GT X Img vs. PredSM X Img
# to do: 2. use multiple discriminators to refine the saliency map from coarse to fine
# In this code, we further add another stream which utilizes the res blocks to extract the semantic information

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
# import torch.nn.functional.tanh as 
import functools
from torch.autograd import Variable
import numpy as np

import time
from torch.autograd import gradcheck

from DCN_lib.modules.modulated_dcn import ModulatedDeformConvPack
from DCN_lib.modules.modulated_dcn import DeformRoIPooling
from DCN_lib.modules.modulated_dcn import ModulatedDeformRoIPoolingPack
from DCN_lib.modules.modulated_dcn import ModulatedDeformConv


###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', gpu_ids=[]):    
    norm_layer = get_norm_layer(norm_type=norm)     
    if netG == 'global':    
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)       
    elif netG == 'local':        
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                  n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    elif netG == 'global_unet':
        netG = GlobalUNet(input_nc, output_nc, ngf)
    else:
        raise('generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)   
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)
'''
class VGGLoss(nn.Module): # The traditional VGG loss
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss
'''
class VGGLoss(nn.Module): # This is actually the L1 loss of predicted Saliency map and Ground-truth density map
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        # self.criterion = KLLoss(gpu_ids) # KL/CC is better than L1 loss to serve as perceptual loss here
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]  
        self.epsilon = 1e-8      

    def forward(self, x, y):              
        # x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        # loss = 0
        x = x.float()
        y = y.float()
        x = x.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols
        y = y.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols

        min1 = torch.min(x)
        max1 = torch.max(x)
        x = (x - min1) / (max1 - min1 + self.epsilon) # min-max normalization for keeping L1 loss non-NAN

        min2 = torch.min(y)
        max2 = torch.max(y)
        y = (y - min2) / (max2 - min2 + self.epsilon) # min-max normalization for keeping L1 loss non-NAN
        
        # L1_loss =  torch.mean( torch.abs(x - y) )
        L1_loss = self.criterion(x, y)
        # for i in range(len(x_vgg)):
          #  loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return L1_loss

class CCLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(CCLoss, self).__init__()        
        # self.vgg = Vgg19().cuda()
        # self.criterion = nn.L1Loss()
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0] 
        self.epsilon = 1e-8


    def forward(self, map_pred, map_gtd):  # map_pred : input prediction saliency map, map_gtd : input ground truth density map
        map_pred = map_pred.float()
        map_gtd = map_gtd.float()
        
        map_pred = map_pred.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols
        map_gtd = map_gtd.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols

        min1 = torch.min(map_pred)
        max1 = torch.max(map_pred)
        map_pred = (map_pred - min1) / (max1 - min1 + self.epsilon) # min-max normalization for keeping KL loss non-NAN

        min2 = torch.min(map_gtd)
        max2 = torch.max(map_gtd)
        map_gtd = (map_gtd - min2) / (max2 - min2 + self.epsilon) # min-max normalization for keeping KL loss non-NAN
        
        map_pred_mean = torch.mean(map_pred) # calculating the mean value of tensor
        map_pred_mean = map_pred_mean.item() # change the tensor into a number

        map_gtd_mean = torch.mean(map_gtd) # calculating the mean value of tensor
        map_gtd_mean = map_gtd_mean.item() # change the tensor into a number
        # print("map_gtd_mean is :", map_gtd_mean)

        map_pred_std = torch.std(map_pred) # calculate the standard deviation
        map_pred_std = map_pred_std.item() # change the tensor into a number 
        map_gtd_std = torch.std(map_gtd) # calculate the standard deviation
        map_gtd_std = map_gtd_std.item() # change the tensor into a number 

        map_pred = (map_pred - map_pred_mean) / (map_pred_std + self.epsilon) # normalization
        map_gtd = (map_gtd - map_gtd_mean) / (map_gtd_std + self.epsilon) # normalization

        map_pred_mean = torch.mean(map_pred) # re-calculating the mean value of normalized tensor
        map_pred_mean = map_pred_mean.item() # change the tensor into a number

        map_gtd_mean = torch.mean(map_gtd) # re-calculating the mean value of normalized tensor
        map_gtd_mean = map_gtd_mean.item() # change the tensor into a number

        CC_1 = torch.sum( (map_pred - map_pred_mean) * (map_gtd - map_gtd_mean) )
        CC_2 = torch.rsqrt(torch.sum(torch.pow(map_pred - map_pred_mean, 2))) * torch.rsqrt(torch.sum(torch.pow(map_gtd - map_gtd_mean, 2))) + self.epsilon
        CC = CC_1 * CC_2
        # print("CC loss is :", CC)
        CC = -CC # the bigger CC, the better



        # we put the L1 loss with CC together for avoiding building a new class
        # L1_loss =  torch.mean( torch.abs(map_pred - map_gtd) )
        # print("CC and L1 are :", CC, L1_loss)
        # CC = CC + L1_loss

        return CC


class KLLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(KLLoss, self).__init__()        
        # self.vgg = Vgg19().cuda()
        # self.criterion = nn.L1Loss()
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0] 
        self.epsilon = 1e-8 # the parameter to make sure the denominator non-zero


    def forward(self, map_pred, map_gtd):  # map_pred : input prediction saliency map, map_gtd : input ground truth density map
        map_pred = map_pred.float()
        map_gtd = map_gtd.float()
        
        map_pred = map_pred.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols
        map_gtd = map_gtd.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols

        min1 = torch.min(map_pred)
        max1 = torch.max(map_pred)
        # print("min1 and max1 are :", min1, max1)
        map_pred = (map_pred - min1) / (max1 - min1 + self.epsilon) # min-max normalization for keeping KL loss non-NAN

        min2 = torch.min(map_gtd)
        max2 = torch.max(map_gtd)
        # print("min2 and max2 are :", min2, max2)
        map_gtd = (map_gtd - min2) / (max2 - min2 + self.epsilon) # min-max normalization for keeping KL loss non-NAN

        map_pred = map_pred / (torch.sum(map_pred) + self.epsilon)# normalization step to make sure that the map_pred sum to 1
        map_gtd = map_gtd / (torch.sum(map_gtd) + self.epsilon) # normalization step to make sure that the map_gtd sum to 1
        # print("map_pred is :", map_pred)
        # print("map_gtd is :", map_gtd)


        KL = torch.log(map_gtd / (map_pred + self.epsilon) + self.epsilon)
        # print("KL 1 is :", KL)
        KL = map_gtd * KL
        # print("KL 2 is :", KL)
        KL = torch.sum(KL)
        # print("KL 3 is :", KL)
        # print("KL loss is :", KL)

        return KL

class NSSLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(NSSLoss, self).__init__()        
        # self.vgg = Vgg19().cuda()
        # self.criterion = nn.L1Loss()
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0] 
        self.epsilon = 1e-8 # the parameter to make sure the denominator non-zero


    def forward(self, map_pred, map_gtd):  # map_pred : input prediction saliency map, map_gtd : input ground truth density map
        map_pred = map_pred.float()
        map_gtd = map_gtd.float()

        map_pred = map_pred.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols
        map_gtd = map_gtd.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols

        min1 = torch.min(map_pred)
        max1 = torch.max(map_pred)
        # print("min1 and max1 are (saliecny map):", min1, max1)
        map_pred = (map_pred - min1) / (max1 - min1 + self.epsilon) # min-max normalization for keeping KL loss non-NAN

        min2 = torch.min(map_gtd)
        max2 = torch.max(map_gtd)
        # print("min2 and max2 are (fixation points) :", min2, max2)
        map_gtd = (map_gtd - min2) / (max2 - min2 + self.epsilon) # min-max normalization for keeping KL loss non-NAN
        
        map_gtd_id_1 = torch.gt(map_gtd, 0.5)
        map_gtd_id_0 = torch.lt(map_gtd, 0.5)
        map_gtd_id_00 = torch.eq(map_gtd, 0.5)
        map_gtd[map_gtd_id_1] = 1.0
        map_gtd[map_gtd_id_0] = 0.0
        map_gtd[map_gtd_id_00] = 0.0

        map_pred_mean = torch.mean(map_pred) # calculating the mean value of tensor
        map_pred_mean = map_pred_mean.item() # change the tensor into a number

        map_pred_std = torch.std(map_pred) # calculate the standard deviation
        map_pred_std = map_pred_std.item() # change the tensor into a number 

        map_pred = (map_pred - map_pred_mean) / (map_pred_std + self.epsilon) # normalization

        NSS = map_pred * map_gtd
        # print("early NSS is :", NSS)
        '''
        dim_NSS = NSS.size()
        print("dim_NSS is :", dim_NSS)
        dim_NSS = dim_NSS[1]
        sum_nss = 0
        dim_sum = 0
        
        for idxnss in range(0, dim_NSS):
            if (NSS[0, idxnss] > 0.05): # # should not be 0, because there are a lot of 0.00XXX in map1_NSS due to float format
                sum_nss += NSS[0, idxnss]
                dim_sum += 1
        
        NSS = sum_nss / dim_sum
        '''
        # NSS = NSS # should not add anythin, because there are a lot of 0.00XXX in map1_NSS due to float format
        # id = torch.nonzero(NSS)
        id = torch.gt(NSS, 0.1) # find out the id of NSS > 0.1
        bignss = NSS[id]
        # print(bignss)
        if(len(bignss) == 0): # NSS[id] is empty 
            id = torch.gt(NSS, -0.00000001) # decrease the threshold, because must set it as tensor not inter
            bignss = NSS[id]
        # NSS = torch.sum(NSS[id])
        # NSS = torch.mean(NSS)
        NSS = torch.mean(bignss)
        
        NSS = -NSS # the bigger NSS the better
        return NSS 
        # return 0 # if return, error : TypeError: mean(): argument 'input' (position 1) must be Tensor, not float

##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model        
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            
            model_downsample_2 = [nn.Conv2d(output_nc, output_nc, kernel_size=4, stride=2, padding=1), 
                                norm_layer(output_nc), 
                                nn.ReLU(True)]

            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]      

            ### final convolution
            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample)) 
            setattr(self, 'model'+str(n)+'_3', nn.Sequential(*model_downsample_2))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1]) # Here, the 1st output_prev is the output feature map of the global generator 
        print("output_prev_global :", output_prev.size())
          
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1): # This code is motivated to design a multi-local-generator architecture, 
                                                                     # although the original paper only use one local generator
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1') 
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')   
            model_downsample_2 = getattr(self, 'model'+str(n_local_enhancers)+'_3')         
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev) # the encoder part of local generator, concat with the final output of global generator, 
                                                                                  # then get into the decoder of the local generator, Notice that it's a direct "add" operation, not concat
            print("output_prev_local :", output_prev.size())
            # output_prev = model_downsample_2(output_prev) # this is my operation, in order to make sure that the resolution of final output saliency map is the same as input image
        
        return output_prev

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        outglobal = self.model(input) 
        print("output_prev_global XXXXXXXXXXXXXXXXXXXXXXX:", outglobal.size())
        return self.model(input)             

class encoderconv_1(nn.Module): # basic conv of encoder
    def __init__(self, in_ch, ou_ch):
        super(encoderconv_1, self).__init__()
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        self.conv = nn.Sequential( 
            nn.Conv2d(in_ch, ou_ch, kernel_size=3, stride=1, padding=1), # output_shape = (image_shape-filter_shape+2*padding)/stride + 1, image_shape is odd number
            # nn.Conv2d(in_ch, ou_ch, kernel_size=4, stride=1, padding=2),
            # nn.BatchNorm2d(ou_ch),
            norm_layer(ou_ch),
            nn.LeakyReLU(0.2), 
        )
    def forward(self, input):
        return self.conv(input)

class encoderconv_2(nn.Module): # basic conv of encoder
    def __init__(self, in_ch, ou_ch):
        super(encoderconv_2, self).__init__()
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        self.conv = nn.Sequential( 
            nn.Conv2d(in_ch, ou_ch, kernel_size=4, stride=2, padding=1), # output_shape = (image_shape-filter_shape+2*padding)/stride + 1, image_shape is odd number
            # nn.BatchNorm2d(ou_ch),
            norm_layer(ou_ch),
            nn.LeakyReLU(0.2), 
        )
    def forward(self, input):
        return self.conv(input)

class decoderconv_1(nn.Module): # basic conv of encoder
    def __init__(self, in_ch, ou_ch):
        super(decoderconv_1, self).__init__()
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)

        self.conv = nn.Sequential( 
            # nn.ConvTranspose2d(in_ch, ou_ch, kernel_size=4, stride=1, padding=1, output_padding=0), 
            nn.ConvTranspose2d(in_ch, ou_ch, kernel_size=3, stride=1, padding=1, output_padding=0), 
            # nn.BatchNorm2d(ou_ch),
            norm_layer(ou_ch),
            nn.ReLU(True), 
        )
    def forward(self, input):
        return self.conv(input)

class decoderconv_2(nn.Module): # basic conv of encoder
    def __init__(self, in_ch, ou_ch):
        super(decoderconv_2, self).__init__()
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)

        self.conv = nn.Sequential( 
            nn.ConvTranspose2d(in_ch, ou_ch, kernel_size=4, stride=2, padding=1, output_padding=0), 
            # nn.BatchNorm2d(ou_ch),
            norm_layer(ou_ch),
            nn.ReLU(True), 
        )
    def forward(self, input):
        return self.conv(input)

class decoderconv_3(nn.Module): # basic conv of encoder
    def __init__(self, in_ch, ou_ch):
        super(decoderconv_3, self).__init__()
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)

        self.conv = nn.Sequential( 
            nn.ConvTranspose2d(in_ch, ou_ch, kernel_size=4, stride=2, padding=1, output_padding=(1,0)), 
            # nn.BatchNorm2d(ou_ch),
            norm_layer(ou_ch),
            nn.ReLU(True), 
        )
    def forward(self, input):
        return self.conv(input)

class dimredconv(nn.Module): # dim-reduction layer, i.e. bottleneck layer
    def __init__(self, in_ch, ou_ch):
        super(dimredconv, self).__init__()

        self.conv = nn.Sequential( 
            nn.Conv2d(in_ch, ou_ch, kernel_size=3, stride=1, padding=1), # output_shape = (image_shape-filter_shape+2*padding)/stride + 1, image_shape is odd number
            # nn.BatchNorm2d(ou_nc),
            nn.Tanh(), 
        )
    def forward(self, input):
        return self.conv(input)

      
class GlobalUNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        super(GlobalUNet, self).__init__()
        
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False) # for resblock
        activation = nn.ReLU(True) # for resblock
        padding_type='reflect' # for resblock
        
        self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        


        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(64, affine=False)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(128, affine=False)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.norm3 = nn.InstanceNorm2d(256, affine=False)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm4 = nn.InstanceNorm2d(512, affine=False)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm5 = nn.InstanceNorm2d(512, affine=False)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.max5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res_1 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_2 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_3 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_4 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        
        self.conv6_1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu6_1 = nn.ReLU(inplace=True)
        self.norm6 = nn.InstanceNorm2d(512, affine=False)




                
        
        self.conv1_1_h = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu1_1_h = nn.ReLU(inplace=True)
        self.conv1_2_h = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.norm1_h = nn.InstanceNorm2d(64, affine=False)
        self.relu1_2_h = nn.ReLU(inplace=True)
        self.max1_h = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2_1_h = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2_1_h = nn.ReLU(inplace=True)
        self.conv2_2_h = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.norm2_h = nn.InstanceNorm2d(128, affine=False)
        self.relu2_2_h = nn.ReLU(inplace=True)
        self.max2_h = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3_1_h = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3_1_h = nn.ReLU(inplace=True)
        self.conv3_2_h = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_2_h = nn.ReLU(inplace=True)
        self.conv3_3_h = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.norm3_h = nn.InstanceNorm2d(256, affine=False)
        self.relu3_3_h = nn.ReLU(inplace=True)
        self.max3_h = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4_1_h = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu4_1_h = nn.ReLU(inplace=True)
        self.conv4_2_h = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu4_2_h = nn.ReLU(inplace=True)
        self.conv4_3_h = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm4_h = nn.InstanceNorm2d(512, affine=False)
        self.relu4_3_h = nn.ReLU(inplace=True)
        self.max4_h = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5_1_h = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_1_h = nn.ReLU(inplace=True)
        self.conv5_2_h = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_2_h = nn.ReLU(inplace=True)
        self.conv5_3_h = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm5_h = nn.InstanceNorm2d(512, affine=False)
        self.relu5_3_h = nn.ReLU(inplace=True)
        self.max5_h = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res_1 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_2 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        
        self.conv6_1_h = nn.ConvTranspose2d(in_channels=512 + 512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.conv6_1_h = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu6_1_h = nn.ReLU(inplace=True)
        self.norm6_h = nn.InstanceNorm2d(512, affine=False)
        
        self.conv7_1_h = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu7_1_h = nn.ReLU(inplace=True)
        self.norm7_h = nn.InstanceNorm2d(128, affine=False)
        
        self.conv8_1_h = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu8_1_h = nn.ReLU(inplace=True)
        self.norm8_h = nn.InstanceNorm2d(3, affine=False)
        
        self.conv9_1_h = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.tanh9_1_h = nn.Tanh()

        '''
        self.dcn_1 = ModulatedDeformConvPack(512, 256, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.dcn_2 = ModulatedDeformConvPack(256, 512, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        # self.dcn_2 = ModulatedDeformConvPack(256, 512, kernel_size=(5,5), stride=1, padding=2, deformable_groups=2, no_bias=True).cuda()
        '''

        # self.dcn_1 = ModulatedDeformConvPack(512, 256, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        # self.dcn_2 = ModulatedDeformConvPack(512, 256, kernel_size=(5,5), stride=1, padding=2, deformable_groups=2, no_bias=True).cuda()
        
        self.onemulone_ird_1_A = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_1_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_1_B = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_1_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_1_C = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.se_1 = SELayer(384, reduction=8)
        self.onemulone_ird_1_D = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)
        self.norm_ird_1 = nn.InstanceNorm2d(256, affine=False)
        self.relu_ird_1 = nn.ReLU(inplace=True)

        self.onemulone_ird_2_A = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_2_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_2_B = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_2_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_2_C = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.se_2 = SELayer(384, reduction=8)
        self.onemulone_ird_2_D = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)
        self.norm_ird_2 = nn.InstanceNorm2d(256, affine=False)
        self.relu_ird_2 = nn.ReLU(inplace=True)

        self.onemulone_ird_3_A = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_3_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_3_B = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_3_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_3_C = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.se_3 = SELayer(384, reduction=8)
        self.onemulone_ird_3_D = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)
        self.norm_ird_3 = nn.InstanceNorm2d(256, affine=False)
        self.relu_ird_3 = nn.ReLU(inplace=True)

        self.onemulone_ird_4_A = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_4_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_4_B = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_4_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_4_C = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.se_4 = SELayer(384, reduction=8)
        self.onemulone_ird_4_D = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)
        self.norm_ird_4 = nn.InstanceNorm2d(256, affine=False)
        self.relu_ird_4 = nn.ReLU(inplace=True)

        self.onemulone_ird_5_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_5_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_5_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_5_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_5_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_5 = SELayer(384, reduction=8)
        self.onemulone_ird_5_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_5 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_5 = nn.ReLU(inplace=True)

        self.onemulone_ird_6_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_6_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_6_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_6_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_6_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_6 = SELayer(384, reduction=8)
        self.onemulone_ird_6_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_6 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_6 = nn.ReLU(inplace=True)

        self.onemulone_ird_7_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_7_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_7_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_7_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_7_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_7 = SELayer(384, reduction=8)
        self.onemulone_ird_7_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_7 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_7 = nn.ReLU(inplace=True)

        self.onemulone_ird_8_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_8_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_8_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_8_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_8_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_8 = SELayer(384, reduction=8)
        self.onemulone_ird_8_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_8 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_8 = nn.ReLU(inplace=True)

        self.onemulone_ird_9_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_9_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_9_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_9_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_9_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_9 = SELayer(384, reduction=8)
        self.onemulone_ird_9_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_9 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_9 = nn.ReLU(inplace=True)

        self.onemulone_ird_10_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_10_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_10_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_10_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_10_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_10 = SELayer(384, reduction=8)
        self.onemulone_ird_10_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_10 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_10 = nn.ReLU(inplace=True)

        self.relu = nn.ReLU(inplace=True)

        
        
        
        
        
        
        
    def forward(self, input):   

        input_small = self.downsample(input)
    
        e1 = self.conv1_1(input_small)
        e1 = self.relu1_1(e1)
        e1 = self.conv1_2(e1)
        e1 = self.norm1(e1)
        e1 = self.relu1_2(e1)
        e1 = self.max1(e1)
        print("e1 size is :", e1.size())
        
        e2 = self.conv2_1(e1)
        e2 = self.relu2_1(e2)
        e2 = self.conv2_2(e2)
        e2 = self.norm2(e2)
        e2 = self.relu2_2(e2)
        e2 = self.max2(e2)
        print("e2 size is :", e2.size())
        
        e3 = self.conv3_1(e2)
        e3 = self.relu3_1(e3)
        e3 = self.conv3_2(e3)
        e3 = self.relu3_2(e3)
        e3 = self.conv3_3(e3)
        e3 = self.norm3(e3)
        e3 = self.relu3_3(e3)
        e3 = self.max3(e3)
        print("e3 size is :", e3.size())
        
        e4 = self.conv4_1(e3)
        e4 = self.relu4_1(e4)
        e4 = self.conv4_2(e4)
        e4 = self.relu4_2(e4)
        e4 = self.conv4_3(e4)
        e4 = self.norm4(e4)
        e4 = self.relu4_3(e4)
        e4 = self.max4(e4)
        print("e4 size is :", e4.size())
        
        e5 = self.conv5_1(e4)
        e5 = self.relu5_1(e5)
        e5 = self.conv5_2(e5)
        e5 = self.relu5_2(e5)
        e5 = self.conv5_3(e5)
        e5 = self.norm5(e5)
        e5 = self.relu5_3(e5)
        e5 = self.max5(e5)
        print("e5 size is :", e5.size())

        res1 = self.res_1(e5)
        res2 = self.res_2(res1)
        res3 = self.res_1(res2)
        res4 = self.res_2(res3)

        e5_upsp = self.conv6_1(res4)
        e5_upsp = self.norm6(e5_upsp)
        e5_upsp = self.relu6_1(e5_upsp)
        # e5_upsp = self.upsp(e5)
        print("e5_upsp size is :", e5_upsp.size())



        
        e1_h = self.conv1_1_h(input)
        # e1_h = self.relu1_1_h(e1_h)
        # e1_h = self.conv1_2_h(e1_h)
        e1_h = self.norm1_h(e1_h)
        e1_h = self.relu1_2_h(e1_h)
        e1_h = self.max1_h(e1_h)
        print("e1_h size is :", e1_h.size())
        
        e2_h = self.conv2_1_h(e1_h)
        # e2_h = self.relu2_1_h(e2_h)
        # e2_h = self.conv2_2_h(e2_h)
        e2_h = self.norm2_h(e2_h)
        e2_h = self.relu2_2_h(e2_h)
        e2_h = self.max2_h(e2_h)
        print("e2_h size is :", e2_h.size())
        
        e3_h = self.conv3_1_h(e2_h)
        # e3_h = self.relu3_1_h(e3_h)
        # e3_h = self.conv3_2_h(e3_h)
        # e3_h = self.relu3_2_h(e3_h)
        # e3_h = self.conv3_3_h(e3_h)
        e3_h = self.norm3_h(e3_h)
        e3_h = self.relu3_3_h(e3_h)
        e3_h = self.max3_h(e3_h)
        print("e3_h size is :", e3_h.size())

        ird_1_A = self.onemulone_ird_1_A(e3_h)
        ird_1_B = self.onemulone_ird_1_B(e3_h)
        ird_1_B = self.dcn_ird_1_B(ird_1_B)
        ird_1_C = self.onemulone_ird_1_C(e3_h)
        ird_1_C = self.dcn_ird_1_C(ird_1_C)
        ird_1_C = self.dcn_ird_1_C(ird_1_C)
        ird_1_concat_ori = torch.cat([ird_1_A, ird_1_B, ird_1_C], dim=1)
        print("size ofird_1_concat_ori is:", ird_1_concat_ori.size())
        ird_1_concat_se = self.se_1(ird_1_concat_ori)
        print("size ofird_1_concat_se is:", ird_1_concat_se.size())
        ird_1_concat_residual = self.onemulone_ird_1_D(ird_1_concat_se)
        # ird_1_concat_residual = self.norm_ird_1(ird_1_concat_residual)
        # ird_1_concat_residual = self.relu_ird_1(ird_1_concat_residual)
        ird_1 = e3_h + ird_1_concat_residual
        print("ird_1 size is:", ird_1.size())
        ird_1 = self.relu(ird_1)

        # ird_1_concat = self.GovalAvgPooling(ird_1_concat)
        # ird_1_concat = torch.mean(ird_1_concat_ori.view(ird_1_concat_ori.size(0), ird_1_concat_ori.size(1), -1), dim=2) # Global pooling
        # print("size of global pooling is:", ird_1_concat.size())
        # ird_1_concat = self.relu_ird_1_1(self.fc_ird_1_1(ird_1_concat))
        # ird_1_concat = self.sigmoid_ird_1_2(self.fc_ird_1_2(ird_1_concat))
        # print("size of SEmodule tensor is:", ird_1_concat, ird_1_concat.size())

        ird_2_A = self.onemulone_ird_2_A(ird_1)
        ird_2_B = self.onemulone_ird_2_B(ird_1)
        ird_2_B = self.dcn_ird_2_B(ird_2_B)
        ird_2_C = self.onemulone_ird_2_C(ird_1)
        ird_2_C = self.dcn_ird_2_C(ird_2_C) # two 3X3 equals one 5X5 deformable conv
        ird_2_C = self.dcn_ird_2_C(ird_2_C)
        ird_2_concat_ori = torch.cat([ird_2_A, ird_2_B, ird_2_C], dim=1)
        print("size ofird_2_concat_ori is:", ird_2_concat_ori.size())
        ird_2_concat_se = self.se_2(ird_2_concat_ori)
        print("size ofird_2_concat_se is:", ird_2_concat_se.size())
        ird_2_concat_residual = self.onemulone_ird_2_D(ird_2_concat_se)
        # ird_2_concat_residual = self.norm_ird_2(ird_2_concat_residual)
        # ird_2_concat_residual = self.relu_ird_2(ird_2_concat_residual)
        ird_2 = ird_1 + ird_2_concat_residual
        print("ird_2 size is:", ird_2.size()) 
        ird_2 = self.relu(ird_2)      

        ird_3_A = self.onemulone_ird_3_A(ird_2)
        ird_3_B = self.onemulone_ird_3_B(ird_2)
        ird_3_B = self.dcn_ird_3_B(ird_3_B)
        ird_3_C = self.onemulone_ird_3_C(ird_2)
        ird_3_C = self.dcn_ird_3_C(ird_3_C) # two 3X3 equals one 5X5 deformable conv
        ird_3_C = self.dcn_ird_3_C(ird_3_C)
        ird_3_concat_ori = torch.cat([ird_3_A, ird_3_B, ird_3_C], dim=1)
        print("size ofird_3_concat_ori is:", ird_3_concat_ori.size())
        ird_3_concat_se = self.se_3(ird_3_concat_ori)
        print("size ofird_3_concat_se is:", ird_3_concat_se.size())
        ird_3_concat_residual = self.onemulone_ird_3_D(ird_3_concat_se)
        # ird_3_concat_residual = self.norm_ird_3(ird_3_concat_residual)
        # ird_3_concat_residual = self.relu_ird_3(ird_3_concat_residual)
        ird_3 = ird_2 + ird_3_concat_residual
        print("ird_3 size is:", ird_3.size())
        ird_3 = self.relu(ird_3)

        ird_4_A = self.onemulone_ird_4_A(ird_3)
        ird_4_B = self.onemulone_ird_4_B(ird_3)
        ird_4_B = self.dcn_ird_4_B(ird_4_B)
        ird_4_C = self.onemulone_ird_4_C(ird_3)
        ird_4_C = self.dcn_ird_4_C(ird_4_C) # two 3X3 equals one 5X5 deformable conv
        ird_4_C = self.dcn_ird_4_C(ird_4_C)
        ird_4_concat_ori = torch.cat([ird_4_A, ird_4_B, ird_4_C], dim=1)
        print("size ofird_4_concat_ori is:", ird_4_concat_ori.size())
        ird_4_concat_se = self.se_4(ird_4_concat_ori)
        print("size ofird_4_concat_se is:", ird_4_concat_se.size())
        ird_4_concat_residual = self.onemulone_ird_4_D(ird_4_concat_se)
        # ird_4_concat_residual = self.norm_ird_4(ird_4_concat_residual)
        # ird_4_concat_residual = self.relu_ird_4(ird_4_concat_residual)
        ird_4 = ird_3 + ird_4_concat_residual
        print("ird_4 size is:", ird_4.size())
        ird_4 = self.relu(ird_4)

        e4_h = self.conv4_1_h(ird_4)
        # e4_h = self.relu4_1_h(e4_h)
        # e4_h = self.conv4_2_h(e4_h)
        # e4_h = self.relu4_2_h(e4_h)
        # e4_h = self.conv4_3_h(e4_h)
        e4_h = self.norm4_h(e4_h)
        e4_h = self.relu4_3_h(e4_h)
        e4_h = self.max4_h(e4_h)
        print("e4_h size is :", e4_h.size())

        ird_5_A = self.onemulone_ird_5_A(e4_h)
        ird_5_B = self.onemulone_ird_5_B(e4_h)
        ird_5_B = self.dcn_ird_5_B(ird_5_B)
        ird_5_C = self.onemulone_ird_5_C(e4_h)
        ird_5_C = self.dcn_ird_5_C(ird_5_C) # two 3X3 equals one 5X5 deformable conv
        ird_5_C = self.dcn_ird_5_C(ird_5_C)
        ird_5_concat_ori = torch.cat([ird_5_A, ird_5_B, ird_5_C], dim=1)
        print("size ofird_5_concat_ori is:", ird_5_concat_ori.size())
        ird_5_concat_se = self.se_5(ird_5_concat_ori)
        print("size ofird_5_concat_se is:", ird_5_concat_se.size())
        ird_5_concat_residual = self.onemulone_ird_5_D(ird_5_concat_se)
        # ird_5_concat_residual = self.norm_ird_5(ird_5_concat_residual)
        # ird_5_concat_residual = self.relu_ird_5(ird_5_concat_residual)
        ird_5 = e4_h + ird_5_concat_residual
        print("ird_5 size is:", ird_5.size())
        ird_5 = self.relu(ird_5)

        ird_6_A = self.onemulone_ird_6_A(ird_5)
        ird_6_B = self.onemulone_ird_6_B(ird_5)
        ird_6_B = self.dcn_ird_6_B(ird_6_B)
        ird_6_C = self.onemulone_ird_6_C(ird_5)
        ird_6_C = self.dcn_ird_6_C(ird_6_C) # two 3X3 equals one 5X5 deformable conv
        ird_6_C = self.dcn_ird_6_C(ird_6_C)
        ird_6_concat_ori = torch.cat([ird_6_A, ird_6_B, ird_6_C], dim=1)
        print("size ofird_6_concat_ori is:", ird_6_concat_ori.size())
        ird_6_concat_se = self.se_6(ird_6_concat_ori)
        print("size ofird_6_concat_se is:", ird_6_concat_se.size())
        ird_6_concat_residual = self.onemulone_ird_6_D(ird_6_concat_se)
        # ird_6_concat_residual = self.norm_ird_6(ird_6_concat_residual)
        # ird_6_concat_residual = self.relu_ird_6(ird_6_concat_residual)
        ird_6 = ird_5 + ird_6_concat_residual
        print("ird_6 size is:", ird_6.size())
        ird_6 = self.relu(ird_6)

        ird_7_A = self.onemulone_ird_7_A(ird_6)
        ird_7_B = self.onemulone_ird_7_B(ird_6)
        ird_7_B = self.dcn_ird_7_B(ird_7_B)
        ird_7_C = self.onemulone_ird_7_C(ird_6)
        ird_7_C = self.dcn_ird_7_C(ird_7_C) # two 3X3 equals one 5X5 deformable conv
        ird_7_C = self.dcn_ird_7_C(ird_7_C)
        ird_7_concat_ori = torch.cat([ird_7_A, ird_7_B, ird_7_C], dim=1)
        print("size ofird_7_concat_ori is:", ird_7_concat_ori.size())
        ird_7_concat_se = self.se_7(ird_7_concat_ori)
        print("size ofird_7_concat_se is:", ird_7_concat_se.size())
        ird_7_concat_residual = self.onemulone_ird_7_D(ird_7_concat_se)
        # ird_7_concat_residual = self.norm_ird_7(ird_7_concat_residual)
        # ird_7_concat_residual = self.relu_ird_7(ird_7_concat_residual)
        ird_7 = ird_6 + ird_7_concat_residual
        print("ird_7 size is:", ird_7.size())
        ird_7 = self.relu(ird_7)

        ird_8_A = self.onemulone_ird_8_A(ird_7)
        ird_8_B = self.onemulone_ird_8_B(ird_7)
        ird_8_B = self.dcn_ird_8_B(ird_8_B)
        ird_8_C = self.onemulone_ird_8_C(ird_7)
        ird_8_C = self.dcn_ird_8_C(ird_8_C) # two 3X3 equals one 5X5 deformable conv
        ird_8_C = self.dcn_ird_8_C(ird_8_C)
        ird_8_concat_ori = torch.cat([ird_8_A, ird_8_B, ird_8_C], dim=1)
        print("size ofird_8_concat_ori is:", ird_8_concat_ori.size())
        ird_8_concat_se = self.se_8(ird_8_concat_ori)
        print("size ofird_8_concat_se is:", ird_8_concat_se.size())
        ird_8_concat_residual = self.onemulone_ird_8_D(ird_8_concat_se)
        # ird_8_concat_residual = self.norm_ird_8(ird_8_concat_residual)
        # ird_8_concat_residual = self.relu_ird_8(ird_8_concat_residual)
        ird_8 = ird_7 + ird_8_concat_residual
        print("ird_8 size is:", ird_8.size())
        ird_8 = self.relu(ird_8)

        e5_h = self.conv5_1_h(ird_8)
        # e5_h = self.relu5_1_h(e5_h)
        # e5_h = self.conv5_2_h(e5_h)
        # e5_h = self.relu5_2_h(e5_h)
        # e5_h = self.conv5_3_h(e5_h)
        e5_h = self.norm5_h(e5_h)
        e5_h = self.relu5_3_h(e5_h)
        e5_h = self.max5_h(e5_h)
        print("e5_h size is :", e5_h.size())

        '''
        e5_h_2 = self.conv_2_h(e4_h)
        e5_h_2 = self.relu5_2_h(e5_h_2)
        e5_h_2 = self.conv5_3_h(e5_h_2)
        e5_h_2 = self.relu5_3_h(e5_h_2)
        
        res1 = self.res_1(e5_h_2)
        res2 = self.res_2(res1)

        e5_h_concat = torch.cat([res2, e5_h], dim=1)
        '''

        
        '''
        ird_9_A = self.onemulone_ird_9_A(e5_h)
        ird_9_B = self.onemulone_ird_9_B(e5_h)
        ird_9_B = self.dcn_ird_9_B(ird_9_B)
        ird_9_C = self.onemulone_ird_9_C(e5_h)
        ird_9_C = self.dcn_ird_9_C(ird_9_C) # two 3X3 equals one 5X5 deformable conv
        ird_9_C = self.dcn_ird_9_C(ird_9_C)
        ird_9_concat_ori = torch.cat([ird_9_A, ird_9_B, ird_9_C], dim=1)
        print("size ofird_9_concat_ori is:", ird_9_concat_ori.size())
        ird_9_concat_se = self.se_9(ird_9_concat_ori)
        print("size ofird_9_concat_se is:", ird_9_concat_se.size())
        ird_9_concat_residual = self.onemulone_ird_9_D(ird_9_concat_se)
        ird_9 = e5_h + ird_9_concat_residual
        print("ird_9 size is:", ird_9.size())
        ird_9 = self.norm_ird_9(ird_9)
        ird_9 = self.relu_ird_9(ird_9)

        ird_10_A = self.onemulone_ird_10_A(ird_9)
        ird_10_B = self.onemulone_ird_10_B(ird_9)
        ird_10_B = self.dcn_ird_10_B(ird_10_B)
        ird_10_C = self.onemulone_ird_10_C(ird_9)
        ird_10_C = self.dcn_ird_10_C(ird_10_C) # two 3X3 equals one 5X5 deformable conv
        ird_10_C = self.dcn_ird_10_C(ird_10_C)
        ird_10_concat_ori = torch.cat([ird_10_A, ird_10_B, ird_10_C], dim=1)
        print("size ofird_10_concat_ori is:", ird_10_concat_ori.size())
        ird_10_concat_se = self.se_10(ird_10_concat_ori)
        print("size ofird_10_concat_se is:", ird_10_concat_se.size())
        ird_10_concat_residual = self.onemulone_ird_10_D(ird_10_concat_se)
        ird_10 = ird_9 + ird_10_concat_residual
        print("ird_10 size is:", ird_10.size())
        ird_10 = self.norm_ird_10(ird_10)
        ird_10 = self.relu_ird_10(ird_10)
        '''
        # res1 = self.res_1(e5_h)
        # res2 = self.res_2(res1)
        
        # e5_h_concat = torch.cat([e5_upsp, e5_h], dim=1)
        # e5_h_concat = torch.cat([res2, res2], dim=1)

        e5_h_concat = torch.cat([e5_upsp, e5_h], dim=1)



        # d1_h = self.conv6_1_h(e5_h)
        d1_h = self.conv6_1_h(e5_h_concat)
        d1_h = self.norm6_h(d1_h)
        d1_h = self.relu6_1_h(d1_h)
        print("d1_h size is :", d1_h.size())
        
        d2_h = self.conv7_1_h(d1_h)
        # d2 = self.norm7(d2)
        d2_h = self.relu7_1_h(d2_h)
        print("d2_h size is :", d2_h.size())
        
        d3_h = self.conv8_1_h(d2_h)
        # d2 = self.norm7(d2)
        d3_h = self.relu8_1_h(d3_h)
        print("d3_h size is :", d3_h.size())
        
        d4_h = self.conv9_1_h(d3_h)
        d4_h = self.tanh9_1_h(d4_h)
        print("d4_h size is :", d4_h.size())
        
        
        return d4_h


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()        
        self.output_nc = output_nc        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                 norm_layer(ngf), nn.ReLU(True)]             
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model) 

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))        
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i)).nonzero() # n x 4            
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]                    
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)                                        
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat                       
        return outputs_mean

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
