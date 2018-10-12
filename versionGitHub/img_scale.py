import cv2
import numpy as np 
import os

big_img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/test/'
small_img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/test_scale/' 
txt_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/'

file = open(txt_path + 'SALICON_5000_name_list.txt', 'w')
for name in os.listdir(big_img_path):
    print(name)
    big_img = cv2.imread(big_img_path + name)
    context = name + '\n'
    file.write(context)
    small_img = cv2.resize(big_img, (320, 240), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(small_img_path + name, small_img)
file.close()
    