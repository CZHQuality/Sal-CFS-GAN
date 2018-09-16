import cv2
import os

read_path = "/media/ubuntu/CZHhy/GAN/QADataBase/TID2013Patch1vs4/ssimDM/train/"
save_path = "/media/ubuntu/CZHhy/GAN/TFlightPix2PixGAN/QA_database/imgs/"
index = 0
for name in os.listdir(read_path):
    index += 1
    print("index = ", index)
    # if(index > 5):
      #  break

    temp = cv2.imread(read_path + name)
    sv_name = name[0:-4]
    cv2.imwrite(save_path + sv_name + '.jpg', temp)