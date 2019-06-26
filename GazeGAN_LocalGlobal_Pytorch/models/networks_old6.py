# We add the saliency losses in this script, which are CC, NSS, KL and L1, and we use L1 loss to replace VGG loss (represented by Class VGGLoss)
# We add the resblock as bottle-neck layer of UNet
# This code is for inference the intermediate representation (feature map) of the pre-trained saliency model
# This code is for constructing the SalGAN network architecture using this whole framework
### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
# import torch.nn.functional.tanh as 
import functools
from torch.autograd import Variable
import numpy as np

################################################################################
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
        # print("cuda ids is :", gpu_ids[0])  
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
        # self.vgg = Vgg19().cuda(gpu_ids[0])
        self.criterion = nn.L1Loss()
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
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1]) # Here, the 1st output_prev is the output feature map of the global generator   
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1): # This code is motivated to design a multi-local-generator architecture, 
                                                                     # although the original paper only use one local generator
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1') 
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev) # the encoder part of local generator, concat with the final output of global generator, 
                                                                                  # then get into the decoder of the local generator, Notice that it's a direct "add" operation, not concat
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

        # encoder of generator
        self.en_conv1 = encoderconv_2(input_nc, 64)
        self.en_conv2 = encoderconv_2(64, 128)
        self.en_conv3 = encoderconv_2(128, 256)
        self.en_conv4 = encoderconv_2(256, 512)
        self.en_conv5 = encoderconv_2(512, 1024)
        self.en_conv6 = encoderconv_2(1024, 1024)
        # self.en_conv7 = encoderconv_2(1024, 1024)
        # self.en_conv8 = encoderconv_2(1024, 1024)
        
        self.res_1 = ResnetBlock(1024, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_2 = ResnetBlock(1024, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_3 = ResnetBlock(1024, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_4 = ResnetBlock(1024, padding_type=padding_type, activation=activation, norm_layer=norm_layer)

        # decoder of generator
        # self.de_conv1 = decoderconv_2(1024, 1024)
        # self.de_conv2 = decoderconv_1(1024+1024, 1024)
        # self.de_conv1 = decoderconv_2(1024, 1024)
        '''
        self.de_conv1 = decoderconv_3(1024, 1024)
        self.de_conv2 = decoderconv_2(1024+1024, 512)
        self.de_conv3 = decoderconv_2(512+512, 256)
        self.de_conv4 = decoderconv_2(256+256, 128)
        self.de_conv5 = decoderconv_2(128+128, 64)
        self.de_conv6 = decoderconv_2(64+64, output_nc)
        '''
        self.de_conv1 = decoderconv_3(1024, 1024)
        self.de_conv2 = decoderconv_2(1024, 512)
        self.de_conv3 = decoderconv_2(512, 256)
        self.de_conv4 = decoderconv_2(256, 128)
        self.de_conv5 = decoderconv_2(128, 64)
        self.de_conv6 = decoderconv_2(64, output_nc)
        # bottle-neck layer
        self.dimr_conv1 = dimredconv(output_nc, output_nc)

    def forward(self, input):
        e1 = self.en_conv1(input)
        # print("size of input is :", input.size())
        # print("size of e1 is :", e1.size())
        e2 = self.en_conv2(e1)
        # print("size of e2 is :", e2.size())
        e3 = self.en_conv3(e2)
        # print("size of e3 is :", e3.size())
        e4 = self.en_conv4(e3)
        # print("size of e4 is :", e4.size())
        e5 = self.en_conv5(e4)
        # print("size of e5 is :", e5.size())
        e6 = self.en_conv6(e5)
        # print("size of e6 is :", e6.size())
        # e7 = self.en_conv7(e6)
        # print("size of e7 is :", e7.size())
        # e8 = self.en_conv8(e7)
        # print("size of e8 is :", e8.size())

        res1 = self.res_1(e6)
        res2 = self.res_2(res1)
        res3 = self.res_2(res2)
        res4 = self.res_2(res3)

        # d1 = self.de_conv1(e6)
        d1 = self.de_conv1(res4)
        # print("d1 and e5 are :", d1.size(), e5.size())
        # d2 = self.de_conv2(torch.cat([d1, e5], dim=1))
        d2 = self.de_conv2(d1)
        # print("d2 and e4 are :", d2.size(), e4.size())
        # d3 = self.de_conv3(torch.cat([d2, e4], dim=1))
        d3 = self.de_conv3(d2)
        # print("d3 and e3 are :", d3.size(), e3.size())
        # d4 = self.de_conv4(torch.cat([d3, e3], dim=1))
        d4 = self.de_conv4(d3)
        # print("d4 and e2 are :", d4.size(), e2.size())
        # d5 = self.de_conv5(torch.cat([d4, e2], dim=1))
        d5 = self.de_conv5(d4)
        # print("d5 and e1 are :", d5.size(), e1.size())
        # d6 = self.de_conv6(torch.cat([d5, e1], dim=1))
        d6 = self.de_conv6(d5)
        
        d7 = self.dimr_conv1(d6)
        
        out = d7 # the real final output
        # out = torch.squeeze(e1, 0)

        # out = e1
        # out = out[0:2, :, :]
        # print("out is :", out, out.size())
        '''
        out1 = d5
        # out = out1[0:1, 0:3, :, :] # this is right
        out = torch.mean(out1, 1) # mean across 64 channel direction
        out = torch.unsqueeze(out, 0)
        print("out1 size is :", out1.size())
        print("out size is :", out.size())
        print("d7 size is :", d7.size())
        '''

        return out

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
