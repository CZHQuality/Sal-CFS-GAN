from __future__ import division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pix2pix import Pix2pix
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # for limiting the debug information of tensorflow
# import cv2

# iters = 200*400 + 2 # taken from pix2pix paper 5.2
# iters = 200*100 + 2 # taken from pix2pix paper 5.2
# iters = 200*10 + 2 # taken from pix2pix paper 5.2
iters = 200*100 + 2 # taken from pix2pix paper 5.2
batch_size = 1 # taken from pix2pix paper 5.2

A = np.load('dataset_y.npy') # original images
B = np.load('dataset_x.npy') # saliency
# B = np.load('dataset_y.npy') # Architectural labels : inverse, from Distortion Map to Distorted Image
# A = np.load('dataset_x.npy') # Photo

with tf.device('/gpu:0'):
    # model = Pix2pix(256, 256, ichan=3, ochan=3)
    # model = Pix2pix(128, 128, ichan=3, ochan=3)
    model = Pix2pix(240, 320, ichan=3, ochan=3)

saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(iters):
        a = np.expand_dims(A[step % A.shape[0]], axis=0)
        b = 2. * np.expand_dims(B[step % B.shape[0]], axis=0) - 1. # normalize because generator use tanh activation in its output layer

        # gloss_curr, dloss_curr, dloss_curr_real, dloss_curr_fake = model.train_step(sess, a, a, b)
        gloss_curr, dloss_curr = model.train_step(sess, a, a, b)
        print('Step %d: G loss: %f | D loss: %f' % (step, gloss_curr, dloss_curr))
        # print('Step %d: real D probability: %f | fake D probability: %f' % (step, dloss_curr_real, dloss_curr_fake))
        # print("real D probability is :", dloss_curr_real)
        # print("fake D probability is :", dloss_curr_fake)

        # if step % 500 == 0:
        # if step % 10000 == 0:
        if step % 5000 == 0:
            fig = plt.figure()
            fig.set_size_inches(10, 10)
            fig.subplots_adjust(left=0, bottom=0,
                                   right=1, top=1, wspace=0, hspace=0.1)
            p = np.random.permutation(B.shape[0])
            for i in range(0, 81, 3):
                
                # Plot 3 images: First is the architectural label, second the generator output, third the ground truth
                fig.add_subplot(9, 9, i + 1)
                plt.imshow(A[p[i // 3]])
                plt.axis('off')
                fig.add_subplot(9, 9, i + 2)
                plt.imshow((model.sample_generator(sess, np.expand_dims(A[p[i // 3]], axis=0), is_training=True)[0] + 1.) / 2.)
                
                plt.axis('off')
                fig.add_subplot(9, 9, i +3)
                plt.imshow(B[p[i // 3]])
                plt.axis('off')
                
                '''
                # Plot 3 images: First is the architectural label, second the generator output, third the ground truth
                fig.add_subplot(9, 9, i + 1)
                # plt.imshow(A[round(p[i // 3])])
                plt.imshow(A[p[i // 3]])
                print("pppppp : ", p[i // 3])
                plt.axis('off')
                fig.add_subplot(9, 9, i + 2)
                # plt.imshow(round((model.sample_generator(sess, np.expand_dims(A[p[i // 3]], axis=0), is_training=True)[0] + 1.) / 2.))
                plt.imshow(A[p[i // 3]])
                print("pppppp : ", p[i // 3])
                plt.axis('off')
                fig.add_subplot(9, 9, i +3)
                #plt.imshow(round(B[p[i // 3]]))
                plt.imshow(A[p[i // 3]])
                print("pppppp : ", p[i // 3])
                plt.axis('off')
                '''
            print("step is :", step)
            plt.show()
            # plt.savefig("images/iter_" + str(step) + ".jpg")
            # plt.savefig("images/iter_%d.jpg" % step)
            # plt.savefig('images/iter_%d.jpg' % (step+1))
            # plt.close()
        
        # Save the trained model 
        # if step % 3000 == 0:
        if ((step % 20000 == 0) and (step != 0)):
            # Save the model
            # save_path = saver.save(sess, "models/model.ckpt")
            # save_path = saver.save(sess, "models/model_inverse.ckpt")
            # save_path = saver.save(sess, "models/model_mse/model.ckpt")
            save_path = saver.save(sess, "models/model_l1_Adv/model.ckpt")
            print("Model saved in file: %s" % save_path)
        
