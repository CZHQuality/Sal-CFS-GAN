# This code is used to generate the resize images of Saliency Database MIT300
# by padding 0 : first the images are resize to 3:4 aspect ratio, the the resized
# images are rescale to [240, 320] for training deep models
# This code is for MIT1003 dataset

import cv2
import numpy as np 
import scipy.misc as misc
from matplotlib import pyplot as plt
import os

root_img_path = 'E:\\Study\\dataset\\mit1003\\mit1003judd\\stimuli\\ALLSTIMULI\\img\\'
root_sm_path = 'E:\\Study\\dataset\\mit1003\\mit1003judd\\fixationmaps\\ALLFIXATIONMAPS\\'
sv_img_path = 'E:\\Study\\dataset\\mit1003\\mit1003judd\\scaled\\image\\'
sv_sm_path = 'E:\\Study\\dataset\\mit1003\\mit1003judd\\scaled\\map\\'

for name in os.listdir(root_img_path):
    # print(name[0:-5])
    temp_img = cv2.imread(root_img_path + name)
    temp_sm = cv2.imread(root_sm_path + name[0:-5] + '_fixMap.jpg')

    img = temp_img
    [height, width, channel] = img.shape 
    # print("height, width, channel are :", height, width, channel)
    aspect_ratio = width / height
    if (aspect_ratio < (4/3)):
        ideal_width = round( height * 4 / 3 )
        padding_radius = round( (ideal_width - width) / 2)
        img_2 = cv2.copyMakeBorder(img, 0, 0, padding_radius, padding_radius, cv2.BORDER_CONSTANT, value=[126, 126, 126])
    
    if (aspect_ratio > (4/3)):
        ideal_height = round(width * 3 / 4 )
        padding_radius = round( (ideal_height - height) / 2)
        img_2 = cv2.copyMakeBorder(img, padding_radius, padding_radius, 0, 0, cv2.BORDER_CONSTANT, value=[126, 126, 126])

    if (aspect_ratio == (4/3)):
        img_2 = img
    
    img_3 = cv2.resize(img_2, (320, 240), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(sv_img_path + name[0:-5] + '.jpg', img_3)

    '''
    img = temp_sm
    [height, width, channel] = img.shape 
    # print("height, width, channel are :", height, width, channel)
    aspect_ratio = width / height
    if (aspect_ratio < (4/3)):
        ideal_width = round( height * 4 / 3 )
        padding_radius = round( (ideal_width - width) / 2)
        img_2 = cv2.copyMakeBorder(img, 0, 0, padding_radius, padding_radius, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    if (aspect_ratio > (4/3)):
        ideal_height = round(width * 3 / 4 )
        padding_radius = round( (ideal_height - height) / 2)
        img_2 = cv2.copyMakeBorder(img, padding_radius, padding_radius, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    if (aspect_ratio == (4/3)):
        img_2 = img
    
    img_3 = cv2.resize(img_2, (320, 240), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(sv_sm_path + name[0:-5] + '_fixMap.png', img_3)
    '''



'''
root_img_path = 'D:\\Paper\\saliency\\Datasets\\BenchmarkIMAGES\\mit300\\BenchmarkIMAGES\\'
root_save_path = 'D:\\Paper\\saliency\\Datasets\\BenchmarkIMAGES\\mit300\\rescaleMIT300\\'
for i in range(1, 301, 1):
    temp_path = root_img_path + 'i' + str(i) + '.jpg' 
    img = cv2.imread(temp_path)
    [height, width, channel] = img.shape 
    # print("height, width, channel are :", height, width, channel)
    aspect_ratio = width / height
    if (aspect_ratio < (4/3)):
        ideal_width = round( height * 4 / 3 )
        padding_radius = round( (ideal_width - width) / 2)
        img_2 = cv2.copyMakeBorder(img, 0, 0, padding_radius, padding_radius, cv2.BORDER_CONSTANT, value=[126, 126, 126])
    
    if (aspect_ratio > (4/3)):
        ideal_height = round(width * 3 / 4 )
        padding_radius = round( (ideal_height - height) / 2)
        img_2 = cv2.copyMakeBorder(img, padding_radius, padding_radius, 0, 0, cv2.BORDER_CONSTANT, value=[126, 126, 126])

    if (aspect_ratio == (4/3)):
        img_2 = img
    
    img_3 = cv2.resize(img_2, (320, 240), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(root_save_path + 'i' + str(i) + '.jpg', img_3)
'''

'''
    plt.figure(0)
    plt.imshow(img_2)
    plt.show() 
'''