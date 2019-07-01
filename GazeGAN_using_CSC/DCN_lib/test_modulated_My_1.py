#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from modules.modulated_dcn import ModulatedDeformConvPack
from modules.modulated_dcn import DeformRoIPooling
from modules.modulated_dcn import ModulatedDeformRoIPoolingPack

from torch.autograd import Variable

deformable_groups = 1
N, inC, inH, inW = 2, 2, 4, 4
outC = 2
kH, kW = 3, 3


def example_dconv():
    from modules.modulated_dcn import ModulatedDeformConv
    # input = torch.randn(2, 64, 128, 128).cuda()
    input = torch.randn((2, 64, 128, 128), requires_grad=True).cuda()
    # input = Variable( torch.randn(2, 64, 128, 128).cuda(), requires_grad=True)

    # wrap all things (offset and mask) in DCN
    dcn = ModulatedDeformConvPack(64, 64, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
    output_1 = dcn(input)
    output_2 = dcn(output_1)
    print("output_2.requires_grad:", output_2.requires_grad)
    output = dcn(output_2)

    print("The hooked gradient information of input:")
    input.register_hook(print) # MyOp: The autograd will calculate all gradient of all leave points as default. 
                               # However, for saving storage memory, these gradients will be released automatically.
                               # Therefore, we could use the "register_hook" function to see the gradient of these released gradient information
                               # Notice that the "register_hook" functions should be put before the "error.backward()" sentence!!!!
    '''
    print("The hooked gradient information of output_1:")
    output_1.register_hook(print)
    print("The hooked gradient information of output_2:")
    output_2.register_hook(print)
    print("The hooked gradient information of output:")
    output.register_hook(print)
    '''

    # output = output_3
    # print("output is:", output, output.shape)
    targert = output.new(*output.size()) # MyOp Constructs a new tensor of the same data type as self tensor.
                                         # The targert is a "fake" and "random" grount-truth label for demo of loss backward.
    # print("target is:", targert, targert.shape)
    targert.data.uniform_(-0.01, 0.01)
    # print("normalized target is:", targert, targert.shape)
    error = (targert - output).mean()
    error.backward()
    print(output.shape)

    
    print("data of input is:", input.data)
    print("gradient of input is:", input.grad)
    print("gradient of output_2 is:", output_2.grad)
    # print("data of gradient of input is:", input.grad.data)
    


if __name__ == '__main__':

    example_dconv()
    # example_dpooling()
    # example_mdpooling()
