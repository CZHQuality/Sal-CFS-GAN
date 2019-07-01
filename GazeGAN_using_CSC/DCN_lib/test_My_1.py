import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules import DeformConv

num_deformable_groups = 2

N, inC, inH, inW = 2, 6, 512, 512
outC, outH, outW = 4, 512, 512
kH, kW = 3, 3

conv = nn.Conv2d(           # this is used to learn the offset from the input (original image or feature maps from the previous layer)
    inC,
    num_deformable_groups * 2 * kH * kW,
    kernel_size=(kH, kW),
    stride=(1, 1),
    padding=(1, 1),
    bias=False).cuda()

conv_offset2d = DeformConv(
    inC,
    outC, (kH, kW),
    stride=1,
    padding=1,
    num_deformable_groups=num_deformable_groups).cuda()


conv_2 = nn.Conv2d(
    4,
    num_deformable_groups * 2 * kH * kW, # My opinion: in this module, the 2 is fixed, which represents the horizontal and vertical offset directions
    kernel_size=(kH, kW),
    stride=(1, 1),
    padding=(1, 1),
    bias=False).cuda()

outC_2 = 8
conv_offset2d_2 = DeformConv(
    4,
    outC_2, (kH, kW),
    stride=1,
    padding=1,
    num_deformable_groups=num_deformable_groups).cuda()


inputs = Variable(torch.randn(N, inC, inH, inW).cuda(), requires_grad=True)
offset = conv(inputs)
#offset = Variable(torch.randn(N, num_deformable_groups * 2 * kH * kW, inH, inW).cuda(), requires_grad=True)
output = conv_offset2d(inputs, offset) # inputs: feed the inputs and the offset into the DeformConv module
# output.backward(output.data)
print(output.size())

offset_2 = conv_2(output)
output_2 = conv_offset2d_2(output, offset_2)
output_2.backward(output_2.data)
print(output_2.size())

