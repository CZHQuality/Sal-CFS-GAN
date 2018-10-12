# This code is used to validate the model performance on Validation Set for model(U_net + ResBlock + Discriminator)
# This code is used to generate the test saliency maps on SALICON Validation Set
from __future__ import division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from pix2pix import Pix2pix
from pix2pix_New import Pix2pix
import os
from scipy import misc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # for limiting the debug information of tensorflow
# import cv2

Dataset_Root_Path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/DataSetNPY/'
root_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/' # Change the "root_path" as your own path


# for test SALICON 5000 validation images , and save both of img, saliency map and ground truth map 
# A = np.load('val_dataset_y_SALICON.npy') # validation set : original images
A = np.load(Dataset_Root_Path + '7_dataset_y_SALICON_val.npy') # validation set : original images XXXXXXXXXXXXXXXXXX
# B = np.load('val_dataset_x_SALICON.npy') # validation set : saliency maps : do not load val_dataset_y_SALICON.npy and val_dataset_x_SALICON.npy together !!!!!
# C = np.load('val_dataset_Pts_SALICON.npy') # validation set : saliency maps
# A = np.load('test_dataset_y_MIT300.npy') # validation set : original images
# A = np.load('train_dataset_y_MIT1003.npy') # validation set : original images
# B = np.load('train_dataset_x_MIT1003.npy') # validation set : saliency maps
# C = np.load('train_dataset_Pts_MIT1003.npy') # validation set : saliency maps
# A = np.load('test_dataset_y_SALICON5000.npy') # validation set : original images
# B = np.load('val_dataset_x_SALICON.npy') # validation set : saliency maps
# A = np.load('val_dataset_y.npy') # validation set : original images
# B = np.load('val_dataset_x.npy') # validation set : saliency maps

with tf.device('/gpu:0'): # build the graph which is the same as the graph build in the training stage
    # model = Pix2pix(256, 256, ichan=3, ochan=3)
    # model = Pix2pix(128, 128, ichan=3, ochan=3)
    model = Pix2pix(240, 320, ichan=3, ochan=3)

saver = tf.train.Saver()


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, "/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/models/model_l1_Adv_3/model.ckpt")
    # saver.restore(sess, "/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/models/model_l1_Adv_4/model.ckpt")
    # saver.restore(sess, "/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/models/model_l1_Adv_5/model2.ckpt")
    saver.restore(sess, root_path + "models/model_l1_Adv_9/model_middle.ckpt") # Change as your own path of trained model
    # saver.restore(sess, root_path + "models/model_l1_Adv_10/model_middle.ckpt")
    # saver.restore(sess, "/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/models/model_l1_Adv_7/model.ckpt")
    print("Model Restored Successfully !!!")

    # a = np.expand_dims(A[step % A.shape[0]], axis=0)
    # b = 2. * np.expand_dims(B[step % B.shape[0]], axis=0) - 1.
    # file = open('/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/MIT_300_name_list.txt', 'r')
    # file = open('/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/MIT_1003_name_list.txt', 'r')
    # file = open('/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/SALICON_5000_val_name_list.txt', 'r')
    file = open(Dataset_Root_Path + '7_SALICON_val_name_list.txt', 'r')  #XXXXXXXXXXXXXXXX
    
    # for i in range(0, 5000, 1):
    for i in range(0, 1999, 1): # XXXXXXXXXXXXXXXXXX
    # for i in range(0, 300, 1):
    # for i in range(0, 1003, 1):
        name = file.readline()
        print("name is :", name)
        print("index = ", i)
        saliency_map = (model.sample_generator(sess, np.expand_dims(A[i], axis=0), is_training=True)[0] + 1.) / 2.
        plt.figure(0)
        # plt.imshow(saliency_map)
        plt.imshow((model.sample_generator(sess, np.expand_dims(A[i], axis=0), is_training=True)[0] + 1.) / 2.)
        # sm_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/SALICON_5000/' + str(i) + '_SM.jpg' # predicted saliency map
        # sm_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/MIT_300/' + str(i) + '_SM.jpg' # predicted saliency map
        # sm_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/MIT_300_Original/' + name[:-5] + '_SM.jpg' # predicted saliency map
        # sm_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/MIT_300_Finetune/' + name[:-5] + '_SM.jpg' # predicted saliency map
        # sm_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/MIT_300_SALICON_10000/' + name[:-5] + '_SM.jpg' # predicted saliency map
        # sm_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/MIT_1003_SALICON_10000/' + name + '_SM.jpg' # predicted saliency map
        # sm_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/MIT_1003_SALICON_10000_2/' + str(i) + '_SM.jpg' # predicted saliency map
        # sm_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/MIT_1003_MIT_1003/' + str(i) + '_SM.jpg' # predicted saliency map
        # sm_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/MIT_1003_MIT_1003/' + str(i) + '_SM.jpg' # predicted saliency map
        # sm_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/SALICON_5000_val_2/' + name + '_SM.jpg' # predicted saliency map
        # sm_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/SALICON_5000_val_3/' + name + '_SM.jpg' # predicted saliency map
        sm_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/SALICON_5000_val_4/' + name + '_SM.jpg' # predicted saliency map
        # sm_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/MIT_300_MIT1003_fromstratch/' + name[:-5] + '_SM.jpg' # predicted saliency map
        misc.imsave(sm_path, saliency_map)
        
        
        ori_img = A[i]
        plt.figure(1)
        # plt.imshow(saliency_map)
        plt.imshow(A[i])
        # im_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/MIT_300/' + str(i) + '_IMG.jpg' # original image
        # im_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/MIT_300_Original/' + name[:-5] + '_IMG.jpg' # original image
        # im_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/MIT_300_Finetune/' + name[:-5] + '_IMG.jpg' # original image
        # im_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/MIT_300_SALICON_10000/' + name[:-5] + '_IMG.jpg' # original image
        # im_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/MIT_1003_SALICON_10000/' + name + '_IMG.jpg' # original image
        # im_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/MIT_1003_SALICON_10000_2/' + str(i) + '_IMG.jpg' # original image
        # im_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/MIT_1003_MIT_1003/' + str(i) + '_IMG.jpg' # original image
        # im_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/SALICON_5000_val_3/' + name + '_IMG.jpg' # original image
        im_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/SALICON_5000_val_4/' + name + '_IMG.jpg' # original image
        # im_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/MIT_300_MIT1003_fromstratch/' + name[:-5] + '_IMG.jpg' # original image
        misc.imsave(im_path, ori_img)
        


        # gt_img = B[i]
        # plt.figure(2)
        # plt.imshow(B[i])
        # im_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/SALICON_5000_val_2/' + name + '_GT.jpg' # ground truth human gaze
        # misc.imsave(im_path, gt_img)


        # gt_Pts = C[i]
        # plt.figure(3)
        # plt.imshow(saliency_map)
        # plt.imshow(C[i])
        # im_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/SALICON/' + str(i) + '_GT.jpg' # ground truth human gaze
        # im_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/MIT_1003_SALICON_10000/' + name + '_GT.jpg' # ground truth human gaze
        # im_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/MIT_1003_MIT_1003/' + str(i) + '_Pts.jpg' # ground truth human gaze
        # im_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/SALICON_5000_val_2/' + name + '_Pts.jpg' # ground truth human gaze
        # misc.imsave(im_path, gt_Pts)

    file.close()    



