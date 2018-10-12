 # This code is used to generated the sorted (from smaller numbers to bigger numbers) namelist of SALICON Validation Set
import os
import sys
import numpy as np
from PIL import Image, ImageOps

img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/val/'
txt_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/'

idx = 0
file = open(txt_path + 'SALICON_5000_val_name_list_sorted.txt', 'w')

filenames = os.listdir(img_path)
filenames.sort(key=lambda x:int(x[-15:-4]))

names = []
# for name in os.listdir(img_path): # not sorted !!!!
for name in filenames: # sorted !!!!
    print(name)
    idx += 1
    # if idx > 50:
      # break
    context = name[:-4] + '\n'
    file.write(context)
    # if name.endswith('.jpg'):
      #  names.append(name[:-4])
file.close()
