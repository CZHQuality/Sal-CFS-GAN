# This code is used to validate the model performance on Validation Set for model(U_net + ResBlock + Discriminator)
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

A = np.load('val_dataset_y_SALICON.npy') # validation set : original images
B = np.load('val_dataset_x_SALICON.npy') # validation set : saliency maps
# A = np.load('val_dataset_y.npy') # validation set : original images
# B = np.load('val_dataset_x.npy') # validation set : saliency maps

with tf.device('/gpu:0'): # build the graph which is the same as the graph build in the training stage
    # model = Pix2pix(256, 256, ichan=3, ochan=3)
    # model = Pix2pix(128, 128, ichan=3, ochan=3)
    model = Pix2pix(240, 320, ichan=3, ochan=3)

saver = tf.train.Saver()


# select 81 images orderly, and generate the saliency maps of these images
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/models/model_l1_Adv_3/model.ckpt")
    print("Model Restored Successfully !!!")

    # a = np.expand_dims(A[step % A.shape[0]], axis=0)
    # b = 2. * np.expand_dims(B[step % B.shape[0]], axis=0) - 1.

    for i in range(0, 5000, 1):
        print("index = ", i)
        saliency_map = (model.sample_generator(sess, np.expand_dims(A[i], axis=0), is_training=True)[0] + 1.) / 2.
        plt.figure(0)
        # plt.imshow(saliency_map)
        plt.imshow((model.sample_generator(sess, np.expand_dims(A[i], axis=0), is_training=True)[0] + 1.) / 2.)
        sm_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/SALICON_5000/' + str(i) + '_SM.jpg' # predicted saliency map
        misc.imsave(sm_path, saliency_map)
        
'''
    # for i in range(0, 5000, 1):
    for i in range(0, 300, 1):
        print("index = ", i)
        saliency_map = (model.sample_generator(sess, np.expand_dims(A[i], axis=0), is_training=True)[0] + 1.) / 2.
        plt.figure(0)
        # plt.imshow(saliency_map)
        plt.imshow((model.sample_generator(sess, np.expand_dims(A[i], axis=0), is_training=True)[0] + 1.) / 2.)
        sm_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/SALICON/' + str(i) + '_SM.jpg' # predicted saliency map
        misc.imsave(sm_path, saliency_map)

        ori_img = A[i]
        plt.figure(1)
        # plt.imshow(saliency_map)
        plt.imshow(A[i])
        im_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/SALICON/' + str(i) + '_IMG.jpg' # original image
        misc.imsave(im_path, ori_img)

        gt_img = B[i]
        plt.figure(2)
        # plt.imshow(saliency_map)
        plt.imshow(B[i])
        im_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/SALICON/' + str(i) + '_GT.jpg' # ground truth human gaze
        misc.imsave(im_path, gt_img)
'''

'''
# This code is used to validate the model performance on Validation Set for model(U_net + Discriminator)
from __future__ import division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pix2pix import Pix2pix
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # for limiting the debug information of tensorflow
# import cv2

A = np.load('val_dataset_y.npy') # validation set : original images
B = np.load('val_dataset_x.npy') # validation set : saliency maps

with tf.device('/gpu:0'): # build the graph which is the same as the graph build in the training stage
    # model = Pix2pix(256, 256, ichan=3, ochan=3)
    # model = Pix2pix(128, 128, ichan=3, ochan=3)
    model = Pix2pix(240, 320, ichan=3, ochan=3)

saver = tf.train.Saver()


# select 81 images orderly, and generate the saliency maps of these images
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/models/model_l1_Adv/model.ckpt")
    print("Model Restored Successfully !!!")

    # a = np.expand_dims(A[step % A.shape[0]], axis=0)
    # b = 2. * np.expand_dims(B[step % B.shape[0]], axis=0) - 1.

    fig = plt.figure()
    fig.set_size_inches(10, 10)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0.1)
    # p = np.random.permutation(B.shape[0]) # p is a randomly sorted vector including many indexs of images/saliency maps

    for i in range(0, 81, 3): # 9 rows, 3 columns, 27 images in total 
    # for i in range(0, 27, 1): # 9 rows, 3 columns, 27 images in total 
                
                # Plot 3 images: First is the architectural label, second the generator output, third the ground truth
                fig.add_subplot(9, 9, i + 1)
                # plt.imshow(A[p[i // 3]])
                plt.imshow(A[i])
                plt.axis('off')
                fig.add_subplot(9, 9, i + 2)
                # plt.imshow((model.sample_generator(sess, np.expand_dims(A[p[i // 3]], axis=0), is_training=True)[0] + 1.) / 2.)
                plt.imshow((model.sample_generator(sess, np.expand_dims(A[i], axis=0), is_training=True)[0] + 1.) / 2.)
                
                plt.axis('off')
                fig.add_subplot(9, 9, i +3)
                # plt.imshow(B[p[i // 3]])
                plt.imshow(B[i])
                plt.axis('off')
    plt.show()
'''



'''
# This code is used to validate the model performance on Validation Set for model(U_net + ResBlock + Discriminator)
from __future__ import division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from pix2pix import Pix2pix
from pix2pix_New import Pix2pix
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # for limiting the debug information of tensorflow
# import cv2

A = np.load('val_dataset_y.npy') # validation set : original images
B = np.load('val_dataset_x.npy') # validation set : saliency maps

with tf.device('/gpu:0'): # build the graph which is the same as the graph build in the training stage
    # model = Pix2pix(256, 256, ichan=3, ochan=3)
    # model = Pix2pix(128, 128, ichan=3, ochan=3)
    model = Pix2pix(240, 320, ichan=3, ochan=3)

saver = tf.train.Saver()


# select 81 images orderly, and generate the saliency maps of these images
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/models/model_l1_Adv_3/model.ckpt")
    print("Model Restored Successfully !!!")

    # a = np.expand_dims(A[step % A.shape[0]], axis=0)
    # b = 2. * np.expand_dims(B[step % B.shape[0]], axis=0) - 1.

    fig = plt.figure()
    fig.set_size_inches(10, 10)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0.1)
    # p = np.random.permutation(B.shape[0]) # p is a randomly sorted vector including many indexs of images/saliency maps

    for i in range(0, 81, 3): # 9 rows, 3 columns, 27 images in total 
    # for i in range(82, 82+81, 3): # 9 rows, 3 columns, 27 images in total 
    # for i in range(0, 27, 1): # 9 rows, 3 columns, 27 images in total 
                
                times = 0 # control image label : 0-81, 82-163, 163-...
                # Plot 3 images: First is the architectural label, second the generator output, third the ground truth
                fig.add_subplot(9, 9, i + 1)
                # plt.imshow(A[p[i // 3]])
                # plt.imshow(A[i])
                plt.imshow(A[i + 81*times])
                plt.axis('off')
                fig.add_subplot(9, 9, i + 2)
                # plt.imshow((model.sample_generator(sess, np.expand_dims(A[p[i // 3]], axis=0), is_training=True)[0] + 1.) / 2.)
                # plt.imshow((model.sample_generator(sess, np.expand_dims(A[i], axis=0), is_training=True)[0] + 1.) / 2.)
                plt.imshow((model.sample_generator(sess, np.expand_dims(A[i + 81*times], axis=0), is_training=True)[0] + 1.) / 2.)
                
                plt.axis('off')
                fig.add_subplot(9, 9, i +3)
                # plt.imshow(B[p[i // 3]])
                # plt.imshow(B[i])
                plt.imshow(B[i + 81*times])
                plt.axis('off')
    plt.show()
'''





