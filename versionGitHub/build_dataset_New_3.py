# This code is used to build the MIT Datasets (MIT1003 and MIT300)
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

'''
# MIT1003 training set
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

    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/train/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/train/'
    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/shan/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/shan/'
    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/val/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/val/'
    img_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/mitdatasets/czhdatasets/mit1003/scaled/image/'
    smap_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/mitdatasets/czhdatasets/mit1003/scaled/map/'
    
    
    idx = 0
    for name in os.listdir(img_path):
        idx += 1
        # if idx > 5000:
        # if idx > 300: #select 300 validation images in total for saving validation time
          #  break
        if name.endswith('.jpg'):
            names.append(name[:-4])
    # print("names are :", names)

    dataset_X = np.zeros((len(names), 240, 320, 3))
    dataset_Y = np.zeros((len(names), 240, 320, 3))
    # dataset_Z = np.zeros((2, 4, 5, 3))
    # print(dataset_Z)

    for i in range(len(names)):
        print(names[i])
        # dataset_X[i] = _image_preprocessing(os.path.join(smap_path, names[i] + '.png'), 320, 240)
        dataset_X[i] = _image_preprocessing(os.path.join(smap_path, names[i] + '_fixMap.png'), 320, 240)
        dataset_Y[i] = _image_preprocessing(os.path.join(img_path, names[i] + '.jpg'), 320, 240)

    # np.save('val_dataset_x.npy', dataset_X) # saliency maps
    # np.save('val_dataset_y.npy', dataset_Y) # original images
    np.save('train_dataset_x_MIT1003.npy', dataset_X) # saliency maps 5000 val images
    np.save('train_dataset_y_MIT1003.npy', dataset_Y) # original images 5000 val images
'''



'''
# MIT300 test set
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

    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/train/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/train/'
    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/shan/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/shan/'
    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/val/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/val/'
    img_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/mitdatasets/czhdatasets/rescaleMIT300/'
    # smap_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/mitdatasets/czhdatasets/mit1003/scaled/map/'
    txt_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/'
    
    idx = 0
    file = open(txt_path + 'MIT_300_name_list.txt', 'w')
    for name in os.listdir(img_path):
        idx += 1
        context = name + '\n'
        file.write(context)
        # if idx > 5000:
        # if idx > 300: #select 300 validation images in total for saving validation time
          #  break
        if name.endswith('.jpg'):
            names.append(name[:-4])
    # print("names are :", names)
    file.close()
    # dataset_X = np.zeros((len(names), 240, 320, 3))
    dataset_Y = np.zeros((len(names), 240, 320, 3))
    # dataset_Z = np.zeros((2, 4, 5, 3))
    # print(dataset_Z)

    for i in range(len(names)):
        print(names[i])
        # dataset_X[i] = _image_preprocessing(os.path.join(smap_path, names[i] + '.png'), 320, 240)
        # dataset_X[i] = _image_preprocessing(os.path.join(smap_path, names[i] + '_fixMap.png'), 320, 240)
        dataset_Y[i] = _image_preprocessing(os.path.join(img_path, names[i] + '.jpg'), 320, 240)

    # np.save('val_dataset_x.npy', dataset_X) # saliency maps
    # np.save('val_dataset_y.npy', dataset_Y) # original images
    # np.save('train_dataset_x_MIT1003.npy', dataset_X) # saliency maps 5000 val images
    # np.save('train_dataset_y_MIT1003.npy', dataset_Y) # original images 5000 val images
    np.save('test_dataset_y_MIT300.npy', dataset_Y) # original images 5000 val images
'''

'''
# SALICON test set 5000
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

    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/train/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/train/'
    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/shan/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/shan/'
    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/val/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/val/'
    # img_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/mitdatasets/czhdatasets/rescaleMIT300/'
    img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/test_scale/'
    # smap_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/mitdatasets/czhdatasets/mit1003/scaled/map/'
    txt_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/'
    
    idx = 0
    # file = open(txt_path + 'MIT_300_name_list.txt', 'w')
    file = open(txt_path + 'SALICON_5000_name_list.txt', 'w')
    for name in os.listdir(img_path):
        idx += 1
        # context = name + '\n'
        context = name[:-4] + '\n'
        
        file.write(context)
        # if idx > 5000:
        # if idx > 300: #select 300 validation images in total for saving validation time
          #  break
        if name.endswith('.jpg'):
            names.append(name[:-4])
    # print("names are :", names)
    file.close()
    # dataset_X = np.zeros((len(names), 240, 320, 3))
    dataset_Y = np.zeros((len(names), 240, 320, 3))
    # dataset_Z = np.zeros((2, 4, 5, 3))
    # print(dataset_Z)

    for i in range(len(names)):
        print(names[i])
        # dataset_X[i] = _image_preprocessing(os.path.join(smap_path, names[i] + '.png'), 320, 240)
        # dataset_X[i] = _image_preprocessing(os.path.join(smap_path, names[i] + '_fixMap.png'), 320, 240)
        dataset_Y[i] = _image_preprocessing(os.path.join(img_path, names[i] + '.jpg'), 320, 240)

    # np.save('val_dataset_x.npy', dataset_X) # saliency maps
    # np.save('val_dataset_y.npy', dataset_Y) # original images
    # np.save('train_dataset_x_MIT1003.npy', dataset_X) # saliency maps 5000 val images
    # np.save('train_dataset_y_MIT1003.npy', dataset_Y) # original images 5000 val images
    # np.save('test_dataset_y_MIT300.npy', dataset_Y) # original images 5000 val images
    np.save('test_dataset_y_SALICON5000.npy', dataset_Y) # original images 5000 val images
'''




'''
# MIT1003 training set with binary discret fixation points
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

    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/train/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/train/'
    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/shan/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/shan/'
    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/val/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/val/'
    img_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/mitdatasets/czhdatasets/mit1003/scaled/image/'
    smap_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/mitdatasets/czhdatasets/mit1003/scaled/map/'
    pts_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/mitdatasets/czhdatasets/mit1003/scaled/pts/'
    
    
    idx = 0
    for name in os.listdir(img_path):
        idx += 1
        # if idx > 5000:
        # if idx > 300: #select 300 validation images in total for saving validation time
          #  break
        if name.endswith('.jpg'):
            names.append(name[:-4])
    # print("names are :", names)

    dataset_X = np.zeros((len(names), 240, 320, 3))
    dataset_Y = np.zeros((len(names), 240, 320, 3))
    dataset_Pts = np.zeros((len(names), 240, 320, 3))
    # dataset_Z = np.zeros((2, 4, 5, 3))
    # print(dataset_Z)

    for i in range(len(names)):
        print(names[i])
        # dataset_X[i] = _image_preprocessing(os.path.join(smap_path, names[i] + '.png'), 320, 240)
        dataset_X[i] = _image_preprocessing(os.path.join(smap_path, names[i] + '_fixMap.png'), 320, 240)        
        dataset_Y[i] = _image_preprocessing(os.path.join(img_path, names[i] + '.jpg'), 320, 240)
        dataset_Pts[i] = _image_preprocessing(os.path.join(pts_path, names[i] + '_fixPts.png'), 320, 240)
    # np.save('val_dataset_x.npy', dataset_X) # saliency maps
    # np.save('val_dataset_y.npy', dataset_Y) # original images
    np.save('train_dataset_x_MIT1003.npy', dataset_X) # saliency maps 5000 val images
    np.save('train_dataset_y_MIT1003.npy', dataset_Y) # original images 5000 val images
    np.save('train_dataset_Pts_MIT1003.npy', dataset_Pts) # discrete binary fixation points image 
'''

'''
# SALICON 5000 validation set with binary discret fixation points
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

    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/train/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/train/'
    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/shan/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/shan/'
    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/val/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/val/'
    # img_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/mitdatasets/czhdatasets/mit1003/scaled/image/'
    # smap_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/mitdatasets/czhdatasets/mit1003/scaled/map/'
    # pts_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/mitdatasets/czhdatasets/mit1003/scaled/pts/'
    img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/val/'
    smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/val/'
    pts_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/fixations_img_scaled/val'
    txt_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/'
    
    idx = 0
    file = open(txt_path + 'SALICON_5000_val_name_list.txt', 'w')
    for name in os.listdir(img_path):
        idx += 1
        # if idx > 5000:
        # if idx > 300: #select 300 validation images in total for saving validation time
          #  break
        context = name[:-4] + '\n'
        file.write(context)
        if name.endswith('.jpg'):
            names.append(name[:-4])    
    file.close()
    # print("names are :", names)
    
    dataset_X = np.zeros((len(names), 240, 320, 3))
    dataset_Y = np.zeros((len(names), 240, 320, 3))
    dataset_Pts = np.zeros((len(names), 240, 320, 3))
    # dataset_Z = np.zeros((2, 4, 5, 3))
    # print(dataset_Z)
   
    for i in range(len(names)):
        print(names[i])
        # dataset_X[i] = _image_preprocessing(os.path.join(smap_path, names[i] + '.png'), 320, 240)
        # dataset_X[i] = _image_preprocessing(os.path.join(smap_path, names[i] + '_fixMap.png'), 320, 240)        
        # dataset_Y[i] = _image_preprocessing(os.path.join(img_path, names[i] + '.jpg'), 320, 240)
        # dataset_Pts[i] = _image_preprocessing(os.path.join(pts_path, names[i] + '_fixPts.png'), 320, 240)
        dataset_X[i] = _image_preprocessing(os.path.join(smap_path, names[i] + '.png'), 320, 240)        
        dataset_Y[i] = _image_preprocessing(os.path.join(img_path, names[i] + '.jpg'), 320, 240)
        dataset_Pts[i] = _image_preprocessing(os.path.join(pts_path, names[i] + '.png'), 320, 240)
    # np.save('val_dataset_x.npy', dataset_X) # saliency maps
    # np.save('val_dataset_y.npy', dataset_Y) # original images
    # np.save('val_dataset_y_SALICON.npy', dataset_X) # saliency maps 5000 val images
    # np.save('val_dataset_y_SALICON.npy', dataset_Y) # original images 5000 val images
    np.save('val_dataset_Pts_SALICON.npy', dataset_Pts) # discrete binary fixation points image 
'''


# SALICON 10000 training set with binary discret fixation points, we divide 2000 images for one DATASET_UNIT (.npy) group, each
# DATASET_UNIT contains 3 .npy files : X_dataset_y (original images), X_dataset_x (ground truth saliency maps), and X_dataset_Pts (ground truth binary fixation maps)
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
    Dataset_Root_Path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/DataSetNPY/'

    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/train/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/train/'
    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/shan/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/shan/'
    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/val/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/val/'
    # img_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/mitdatasets/czhdatasets/mit1003/scaled/image/'
    # smap_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/mitdatasets/czhdatasets/mit1003/scaled/map/'
    # pts_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/mitdatasets/czhdatasets/mit1003/scaled/pts/'
    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/val/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/val/'
    # pts_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/fixations_img_scaled/val'
    # txt_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/'
    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/train/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/train/'
    # pts_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/fixations_img_scaled/train/'
    img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/val/'
    smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/val/'
    pts_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/fixations_img_scaled/val/'
    # txt_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/'
    txt_path = Dataset_Root_Path
    
    idx = 0
    # file = open(txt_path + 'SALICON_5000_val_name_list.txt', 'w')
    # file = open(txt_path + '6_SALICON_train_name_list.txt', 'w') #XXXXXXXXXXXXXXXXXXXXXXXX
    file = open(txt_path + '9_SALICON_val_name_list.txt', 'w') #XXXXXXXXXXXXXXXXXXXXXXXX

    filenames = os.listdir(img_path)
    filenames.sort(key=lambda x:int(x[-15:-4]))

    # for name in os.listdir(img_path):
    for name in filenames:
        idx += 1
        print(idx, name)
        # if idx > 5000:
        if idx < 4001: #XXXXXXXXXXXXXXXXXXXXXX
            continue
        if idx > 5000: # select 2000 images to generate the first UNIT_DATASET  XXXXXXXXXXXXXXXXXXXXXXXXXXX
            break
        context = name[:-4] + '\n'
        file.write(context)
        if name.endswith('.jpg'):
            names.append(name[:-4])    
    file.close()
    # print("names are :", names)
    
    dataset_X = np.zeros((len(names), 240, 320, 3))
    dataset_Y = np.zeros((len(names), 240, 320, 3))
    dataset_Pts = np.zeros((len(names), 240, 320, 3))
    # dataset_Z = np.zeros((2, 4, 5, 3))
    # print(dataset_Z)
   
    for i in range(len(names)):
        print(i, names[i])
        # dataset_X[i] = _image_preprocessing(os.path.join(smap_path, names[i] + '.png'), 320, 240)
        # dataset_X[i] = _image_preprocessing(os.path.join(smap_path, names[i] + '_fixMap.png'), 320, 240)        
        # dataset_Y[i] = _image_preprocessing(os.path.join(img_path, names[i] + '.jpg'), 320, 240)
        # dataset_Pts[i] = _image_preprocessing(os.path.join(pts_path, names[i] + '_fixPts.png'), 320, 240)
        dataset_X[i] = _image_preprocessing(os.path.join(smap_path, names[i] + '.png'), 320, 240)        
        dataset_Y[i] = _image_preprocessing(os.path.join(img_path, names[i] + '.jpg'), 320, 240)
        dataset_Pts[i] = _image_preprocessing(os.path.join(pts_path, names[i] + '.png'), 320, 240)
    # np.save('val_dataset_x.npy', dataset_X) # saliency maps
    # np.save('val_dataset_y.npy', dataset_Y) # original images
    # np.save(Dataset_Root_Path + '6_dataset_x_SALICON_train.npy', dataset_X) # XXXXXXXXXXXXXXXXX saliency maps 5000 val images
    # np.save(Dataset_Root_Path + '6_dataset_y_SALICON_train.npy', dataset_Y) # XXXXXXXXXXXXXXXXXXXXX original images 5000 val images
    # np.save(Dataset_Root_Path + '6_dataset_Pts_SALICON_train.npy', dataset_Pts) # XXXXXXXXXXXXXXXXXXXXXX discrete binary fixation points image 
    np.save(Dataset_Root_Path + '9_dataset_x_SALICON_val.npy', dataset_X) # XXXXXXXXXXXXXXXXX saliency maps 5000 val images
    np.save(Dataset_Root_Path + '9_dataset_y_SALICON_val.npy', dataset_Y) # XXXXXXXXXXXXXXXXXXXXX original images 5000 val images
    np.save(Dataset_Root_Path + '9_dataset_Pts_SALICON_val.npy', dataset_Pts) # XXXXXXXXXXXXXXXXXXXXXX discrete binary fixation points image 




'''
# MIT1003 dataset (for generate the "name list txt file for MIT1003")
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

    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/train/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/train/'
    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/shan/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/shan/'
    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/val/'
    # smap_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/maps/val/'
    # img_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/mitdatasets/czhdatasets/rescaleMIT300/'
    # img_path = '/media/ubuntu/CZHhy/Saliency/SALICON/salicon_data/images/test_scale/'
    img_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/mitdatasets/czhdatasets/mit1003/scaled/image/'
    # smap_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/mitdatasets/czhdatasets/mit1003/scaled/map/'
    txt_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/'
    
    idx = 0
    # file = open(txt_path + 'MIT_300_name_list.txt', 'w')
    # file = open(txt_path + 'SALICON_5000_name_list.txt', 'w')
    file = open(txt_path + 'MIT_1003_name_list.txt', 'w')
    for name in os.listdir(img_path):
        idx += 1
        # context = name + '\n'
        context = name[:-4] + '\n'
        
        file.write(context)
        # if idx > 5000:
        # if idx > 300: #select 300 validation images in total for saving validation time
          #  break
        if name.endswith('.jpg'):
            names.append(name[:-4])
    # print("names are :", names)
    file.close()
    # dataset_X = np.zeros((len(names), 240, 320, 3))
    dataset_Y = np.zeros((len(names), 240, 320, 3))
    # dataset_Z = np.zeros((2, 4, 5, 3))
    # print(dataset_Z)

    for i in range(len(names)):
        print(names[i])
        # dataset_X[i] = _image_preprocessing(os.path.join(smap_path, names[i] + '.png'), 320, 240)
        # dataset_X[i] = _image_preprocessing(os.path.join(smap_path, names[i] + '_fixMap.png'), 320, 240)
        # dataset_Y[i] = _image_preprocessing(os.path.join(img_path, names[i] + '.jpg'), 320, 240)

    # np.save('val_dataset_x.npy', dataset_X) # saliency maps
    # np.save('val_dataset_y.npy', dataset_Y) # original images
    # np.save('train_dataset_x_MIT1003.npy', dataset_X) # saliency maps 5000 val images
    # np.save('train_dataset_y_MIT1003.npy', dataset_Y) # original images 5000 val images
    # np.save('test_dataset_y_MIT300.npy', dataset_Y) # original images 5000 val images
    # np.save('test_dataset_y_SALICON5000.npy', dataset_Y) # original images 5000 val images
'''
