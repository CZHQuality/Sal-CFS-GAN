# This code is used to test the performance of the Histogram Estimation Method
# Please refer to "Learning Deep Embeddings with Histogram Loss" for more details
# Differentiable Histogram Loss
# When image is in int8 format, it's a special case of Differentiable Histogram Loss

import cv2
import tensorflow as tf 
import numpy as np 
from matplotlib import pyplot as plt

###########################################
# Defualt Histogram Counting Method of tensorflow
root_path = 'D:\\Paper\\saliency\\Datasets\\SaliencyResult\\SALICON_5000_val_2\\' # Change as your own path
path_1 = root_path + 'COCO_val2014_000000000133__GT.jpg' # GT 
path_2 = root_path + 'COCO_val2014_000000000133__SM.jpg' # Predicted SM

img_GT = tf.read_file(path_1)
img_GT = tf.image.decode_jpeg(img_GT, channels=1) # grayscale format
# img_GT = tf.cast(img_GT, tf.int32) # shape of img_GT is : [240 320   1]
img_GT = tf.to_float(img_GT, name='ToFloat')

# img_SM = tf.read_file(path_1)
img_SM = tf.read_file(path_2)
img_SM = tf.image.decode_jpeg(img_SM, channels=1) # grayscale format
# img_SM = tf.cast(img_SM, tf.int32)
img_SM = tf.to_float(img_SM, name='ToFloat')

# print("1 GT is:", img_GT)
# img_GT = tf.convert_to_tensor(img_GT)
# print("2 GT is:", img_GT)


# da_1 = tf.argmax(img_GT)
# da_1 = tf.argmax(da_1)
# sp_1 = img_GT.get_shape()
sp_1 = tf.shape(img_GT)
da_1_img = tf.reduce_max(img_GT)
# xiao_1 = tf.reduce_min(img_GT)

sp_2 = tf.shape(img_SM)
# da_2 = tf.reduce_max(img_SM)
# xiao_2 = tf.reduce_min(img_SM)

nbins = 100 #256
# VALUE_RANGE = [0, 255]
VALUE_RANGE = [0.0, 255.0]

hist_GT = tf.histogram_fixed_width(img_GT, VALUE_RANGE, nbins)

hist_SM = tf.histogram_fixed_width(img_SM, VALUE_RANGE, nbins)



#####################################
# Differentiable Histogram Counting Method
hist_GT_Dev = np.zeros([1, nbins])

delta = 255 / nbins

BIN_Table = np.arange(0,100,1)
BIN_Table = BIN_Table.astype(np.float64)
BIN_Table = BIN_Table * delta
S_total_pixels = 240 * 320 # The rows and columns of the input image

# BIN_Table_2 = np.zeros([1, nbins])

with tf.Session() as sess:
    img_GT_2 = img_GT.eval(session=sess) # change Tensor into Array


for dim in range(1, 99, 1):
    h_r = BIN_Table[dim] # h_r
    h_r_sub_1 = BIN_Table[dim-1] # h_(r-1)
    h_r_plus_1 = BIN_Table[dim+1] # h_(r+1)
    
    for row in range(0, 240):
        for col in range(0, 320):
            Pixel = img_GT_2[row, col]
            if((Pixel >= h_r_sub_1) and (Pixel < h_r)):
                hist_GT_Dev[0, dim] = hist_GT_Dev[0, dim] + (Pixel - h_r_sub_1) / (S_total_pixels * delta)
            if((Pixel >= h_r) and (Pixel < h_r_plus_1)):
                hist_GT_Dev[0, dim] = hist_GT_Dev[0, dim] + (h_r_plus_1 - Pixel) / (S_total_pixels * delta)

with tf.Session() as sess:
    print("The hist of GT is :", sess.run(hist_GT)) # Traditional Histogram Counting Method
    # print("The hist of SM is :", sess.run(hist_SM))

    # print("One of the gary values of GT is :", sess.run(img_GT[100,200,0]))
    # print("One of the gary values of GT is :", sess.run(img_GT[240-1,320-1,0]))
    # print(hist_GT_Dev)
    print("BIN_Table_2 is ", hist_GT_Dev) # The Differentiable Histogram Counting Method
    # print(h_r)
  
    
