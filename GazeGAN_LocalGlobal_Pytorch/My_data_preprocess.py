# This code is used to divide the Cityspaces dataset into image, label, instance folders respectively
# for preprocessing the real images 
'''
import os
from PIL import Image

root_path = 'D:\\Study\\code\\pix2pixHD-master\\datasets\\leftImg8bit_trainvaltest\\leftImg8bit\\train\\'
subfolders = os.listdir(root_path)
root_path_2 = 'D:\\Study\\code\\pix2pixHD-master\\pix2pixHD\\datasets\\cityscapes\\train_img\\'
# print(subfolders)
len_1 = len(subfolders)

for i in range(1): # len_1
    temp_path = root_path + subfolders[i] + '\\'
    print("subfolders", subfolders[i], temp_path)
    # sv_path = root_path + subfolders[i]
    names = os.listdir(temp_path)
    len_2 = len(names)
    for j in range(len_2):
        temp_path_2 = temp_path + names[j]
        sv_path = root_path_2 + names[j]
        img = Image.open(temp_path_2)
        img.save(sv_path)
'''

import os
from PIL import Image

root_path = 'D:\\Study\\code\\pix2pixHD-master\\datasets\\gtFine_trainvaltest\\gtFine\\train\\'
subfolders = os.listdir(root_path)
root_path_inst = 'D:\\Study\\code\\pix2pixHD-master\\pix2pixHD\\datasets\\cityscapes\\train_inst\\'
root_path_label = 'D:\\Study\\code\\pix2pixHD-master\\pix2pixHD\\datasets\\cityscapes\\train_label\\'
# print(subfolders)
len_1 = len(subfolders)

for i in range(len_1): # len_1
    temp_path = root_path + subfolders[i] + '\\'
    print("subfolders", subfolders[i], temp_path)
    # sv_path = root_path + subfolders[i]
    names = os.listdir(temp_path)
    len_2 = len(names)
    for j in range(len_2):
        name = names[j]
        # print(name)
        # print(name[-15:-4])
        
        if(name[-15:-4] == 'instanceIds'):
            # print(name[-15:-4])
            print(name)
            img = Image.open(temp_path + name)
            img.save(root_path_inst + name)
        
        
        if(name[-12:-4] == 'labelIds'):
            # print(name[-15:-4])
            print(name)
            img = Image.open(temp_path + name)
            img.save(root_path_label + name)
        



