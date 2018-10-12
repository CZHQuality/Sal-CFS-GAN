import os
import sys
import numpy as np
from PIL import Image, ImageOps
'''
def _image_preprocessing(filename, xsize, ysize):
    im = Image.open(filename)

    if filename.endswith('.png'):
        im = im.convert('RGB')
    downsampled_im = ImageOps.fit(im, (xsize, ysize), method=Image.LANCZOS)
    norm_im = np.array(downsampled_im, dtype=np.float32) / 255.

    downsampled_im.close()
    im.close()
    return norm_im

if __name__ == '__main__':
    names = []

    for name in os.listdir(sys.argv[1]):
        if name.endswith('.jpg'):
            names.append(name[:-4])
    # print("names are :", names)

    dataset_X = np.zeros((len(names), 128, 128, 3))
    dataset_Y = np.zeros((len(names), 128, 128, 3))
    # dataset_Z = np.zeros((2, 4, 5, 3))
    # print(dataset_Z)

    for i in range(len(names)):
        print(names[i])
        dataset_X[i] = _image_preprocessing(os.path.join(sys.argv[1], names[i] + '.jpg'), 128, 128)
        dataset_Y[i] = _image_preprocessing(os.path.join(sys.argv[1], names[i] + '.png'), 128, 128)

    np.save('dataset_x.npy', dataset_X)
    np.save('dataset_y.npy', dataset_Y)
'''


def _image_preprocessing(filename, xsize, ysize):
    im = Image.open(filename)
    
    # if filename.endswith('.jpg'):
    if filename.endswith('.png'):
        im = im.convert('RGB')
    if filename.endswith('.jpg'):
        im = im.convert('RGB')
    downsampled_im = ImageOps.fit(im, (xsize, ysize), method=Image.LANCZOS)
    norm_im = np.array(downsampled_im, dtype=np.float32) / 255.

    downsampled_im.close()
    im.close()
    return norm_im

if __name__ == '__main__':
    names = []

    img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/train/'
    smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/train/'
    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/shan/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/shan/'
    '''
    #This is used to generate the first 5000 images
    idx = 0
    for name in os.listdir(img_path):
        idx += 1
        if idx > 5000:
            break
        if name.endswith('.jpg'):
            names.append(name[:-4])
    # print("names are :", names)
    '''

    #This is used to generate the last 5000 images
    idx = 0
    for name in os.listdir(img_path):
        idx += 1
        # print("index is :", idx)
        if idx < 5000:
            continue
        if idx > 10000:
            break
        if name.endswith('.jpg'):
            names.append(name[:-4])
    # print("names are :", names)

    dataset_X = np.zeros((len(names), 240, 320, 3))
    dataset_Y = np.zeros((len(names), 240, 320, 3))
    # dataset_Z = np.zeros((2, 4, 5, 3))
    # print(dataset_Z)

    for i in range(len(names)):
        print(names[i])
        dataset_X[i] = _image_preprocessing(os.path.join(smap_path, names[i] + '.png'), 320, 240)
        dataset_Y[i] = _image_preprocessing(os.path.join(img_path, names[i] + '.jpg'), 320, 240)

    # np.save('dataset_x_1.npy', dataset_X) # saliency maps (1-5000)
    # np.save('dataset_y_1.npy', dataset_Y) # original images (1-5000)
    np.save('dataset_x_2.npy', dataset_X) # saliency maps (5001-10000)
    np.save('dataset_y_2.npy', dataset_Y) # original images (5001-10000)

 
