
# Use NSS+CC+KLD loss, add another placeholder (fixation points map) 
from __future__ import division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from pix2pix import Pix2pix
from pix2pix_New import Pix2pix
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # for limiting the debug information of tensorflow
# import cv2

root_path = '/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/' # Change the "root_path" as your own path
# iters = 200*100 + 2 # taken from pix2pix paper 5.2
# iters = 200*100 + 2 # taken from pix2pix paper 5.2
# iters = 200*10 + 2 # taken from pix2pix paper 5.2
iters = 5000 + 2 # taken from pix2pix paper 5.2
# iters = 400*100 + 2 # taken from pix2pix paper 5.2
batch_size = 1 # taken from pix2pix paper 5.2

# A = np.load('dataset_y.npy') # original images
# B = np.load('dataset_x.npy') # saliency
# A = np.load('val_dataset_y.npy') # original images
# B = np.load('val_dataset_x.npy') # saliency
# A = np.load('dataset_y_1.npy') # original images (SALICON 1-5000)
# B = np.load('dataset_x_1.npy') # saliency
# A = np.load('dataset_y_2.npy') # original images (SALICON 5001-10000)
# B = np.load('dataset_x_2.npy') # saliency
# A = np.load('val_dataset_y_1.npy') # original images
# B = np.load('val_dataset_x_1.npy') # saliency
A = np.load(root_path + 'train_dataset_y_MIT1003.npy') # original images 
B = np.load(root_path + 'train_dataset_x_MIT1003.npy') # saliency maps
C = np.load(root_path + 'train_dataset_Pts_MIT1003.npy') # fixation points maps

with tf.device('/gpu:0'):
    # model = Pix2pix(256, 256, ichan=3, ochan=3)
    # model = Pix2pix(128, 128, ichan=3, ochan=3)
    model = Pix2pix(240, 320, ichan=3, ochan=3)

saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, root_path + "models/model_l1_Adv_3/model.ckpt") # for finetune, not train from stratch
    saver.restore(sess, root_path + "models/model_l1_Adv_5/model2.ckpt") # for finetune, not train from stratch
    print("Model Restored Successfully !!!")
    for step in range(iters):
        a = np.expand_dims(A[step % A.shape[0]], axis=0)
        b = 2. * np.expand_dims(B[step % B.shape[0]], axis=0) - 1. # normalize because generator use tanh activation in its output layer
        c = np.expand_dims(C[step % C.shape[0]], axis=0) # fixation points

        # gloss_curr, dloss_curr, dloss_curr_real, dloss_curr_fake = model.train_step(sess, a, a, b)
        gloss_curr, dloss_curr, monitor = model.train_step(sess, a, a, b, c)
        print('Step %d: G loss: %f | D loss: %f' % (step, gloss_curr, dloss_curr))
        print("monitor (KLD, CC and NSS and Hist-ACS scores):", monitor)
        # print('Step %d: G loss: %f | D loss: %f | monitor loss: %f' % (step, gloss_curr, dloss_curr, monitor))
        # print('Step %d: real D probability: %f | fake D probability: %f' % (step, dloss_curr_real, dloss_curr_fake))
        # print("real D probability is :", dloss_curr_real)
        # print("fake D probability is :", dloss_curr_fake)

        # if step % 1000 == 0:
        if step % 5000 == 0:
        # if step % 10000 == 0:
        # if step % 10000 == 0:
            fig = plt.figure()
            fig.set_size_inches(10, 10)
            fig.subplots_adjust(left=0, bottom=0,
                                   right=1, top=1, wspace=0, hspace=0.1)
            p = np.random.permutation(B.shape[0])
            for i in range(0, 81, 3):
                
                # Plot 3 images: First is the input image, second the generator output, third the ground truth
                fig.add_subplot(9, 9, i + 1)
                plt.imshow(A[p[i // 3]])
                plt.axis('off')
                fig.add_subplot(9, 9, i + 2)
                plt.imshow((model.sample_generator(sess, np.expand_dims(A[p[i // 3]], axis=0), is_training=True)[0] + 1.) / 2.)
                
                plt.axis('off')
                fig.add_subplot(9, 9, i +3)
                plt.imshow(B[p[i // 3]])
                plt.axis('off')
            print("step is :", step)
            plt.show()
            # plt.savefig("images/iter_" + str(step) + ".jpg")
            # plt.savefig("images/iter_%d.jpg" % step)
            # plt.savefig('images/iter_%d.jpg' % (step+1))
            # plt.close()
        
        # Save the trained model 
        # if step % 3000 == 0:
        
        # if ((step % 20000 == 0) and (step != 0)):
        
        # if ((step % 40000 == 0) and (step != 0)):
        '''
        # if ((step % 20000 == 0) and (step != 0)):
        if ((step % 5000 == 0) and (step != 0)):
            # Save the model
            # save_path = saver.save(sess, "models/model.ckpt")
            # save_path = saver.save(sess, "models/model_inverse.ckpt")
            # save_path = saver.save(sess, "models/model_mse/model.ckpt")
            # save_path = saver.save(sess, "models/model_l1_Adv_2/model.ckpt")
            # save_path = saver.save(sess, "models/model_l1_Adv_5/model.ckpt")
            # save_path = saver.save(sess, "models/model_l1_Adv_5/model2.ckpt")
            # save_path = saver.save(sess, root_path + "models/model_l1_Adv_7/model.ckpt")
            save_path = saver.save(sess, root_path + "models/model_l1_Adv_8/model.ckpt")
            print("Model saved in file: %s" % save_path)
        '''


'''
# Use the 300 val images to train the model 
from __future__ import division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from pix2pix import Pix2pix
from pix2pix_New import Pix2pix
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # for limiting the debug information of tensorflow
# import cv2

iters = 200*100 + 2 # taken from pix2pix paper 5.2
# iters = 200*100 + 2 # taken from pix2pix paper 5.2
# iters = 200*10 + 2 # taken from pix2pix paper 5.2
# iters = 500 + 2 # taken from pix2pix paper 5.2
# iters = 400*100 + 2 # taken from pix2pix paper 5.2
batch_size = 1 # taken from pix2pix paper 5.2

# A = np.load('dataset_y.npy') # original images
# B = np.load('dataset_x.npy') # saliency
# A = np.load('val_dataset_y.npy') # original images
# B = np.load('val_dataset_x.npy') # saliency
# A = np.load('dataset_y_1.npy') # original images (SALICON 1-5000)
# B = np.load('dataset_x_1.npy') # saliency
A = np.load('dataset_y_2.npy') # original images (SALICON 5001-10000)
B = np.load('dataset_x_2.npy') # saliency
# A = np.load('val_dataset_y_1.npy') # original images
# B = np.load('val_dataset_x_1.npy') # saliency

with tf.device('/gpu:0'):
    # model = Pix2pix(256, 256, ichan=3, ochan=3)
    # model = Pix2pix(128, 128, ichan=3, ochan=3)
    model = Pix2pix(240, 320, ichan=3, ochan=3)

saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, "/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/models/model_l1_Adv_3/model.ckpt") # for finetune, not train from stratch
    saver.restore(sess, "/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/models/model_l1_Adv_5/model.ckpt") # for finetune, not train from stratch
    print("Model Restored Successfully !!!")
    for step in range(iters):
        a = np.expand_dims(A[step % A.shape[0]], axis=0)
        b = 2. * np.expand_dims(B[step % B.shape[0]], axis=0) - 1. # normalize because generator use tanh activation in its output layer

        # gloss_curr, dloss_curr, dloss_curr_real, dloss_curr_fake = model.train_step(sess, a, a, b)
        gloss_curr, dloss_curr, monitor = model.train_step(sess, a, a, b)
        print('Step %d: G loss: %f | D loss: %f' % (step, gloss_curr, dloss_curr))
        print("monitor :", monitor)
        # print('Step %d: G loss: %f | D loss: %f | monitor loss: %f' % (step, gloss_curr, dloss_curr, monitor))
        # print('Step %d: real D probability: %f | fake D probability: %f' % (step, dloss_curr_real, dloss_curr_fake))
        # print("real D probability is :", dloss_curr_real)
        # print("fake D probability is :", dloss_curr_fake)

        # if step % 500 == 0:
        # if step % 10000 == 0:
        if step % 10000 == 0:
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
            print("step is :", step)
            plt.show()
            # plt.savefig("images/iter_" + str(step) + ".jpg")
            # plt.savefig("images/iter_%d.jpg" % step)
            # plt.savefig('images/iter_%d.jpg' % (step+1))
            # plt.close()
        
        # Save the trained model 
        # if step % 3000 == 0:
        
        # if ((step % 20000 == 0) and (step != 0)):
        
        # if ((step % 40000 == 0) and (step != 0)):
        if ((step % 20000 == 0) and (step != 0)):
            # Save the model
            # save_path = saver.save(sess, "models/model.ckpt")
            # save_path = saver.save(sess, "models/model_inverse.ckpt")
            # save_path = saver.save(sess, "models/model_mse/model.ckpt")
            # save_path = saver.save(sess, "models/model_l1_Adv_2/model.ckpt")
            # save_path = saver.save(sess, "models/model_l1_Adv_5/model.ckpt")
            save_path = saver.save(sess, "models/model_l1_Adv_5/model2.ckpt")
            print("Model saved in file: %s" % save_path)
'''        
              
        




'''
# Use the 1-5000 training images to train the model 
from __future__ import division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from pix2pix import Pix2pix
from pix2pix_New import Pix2pix
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # for limiting the debug information of tensorflow
# import cv2

# iters = 200*400 + 2 # taken from pix2pix paper 5.2
# iters = 200*100 + 2 # taken from pix2pix paper 5.2
# iters = 200*10 + 2 # taken from pix2pix paper 5.2
# iters = 500 + 2 # taken from pix2pix paper 5.2
iters = 400*100 + 2 # taken from pix2pix paper 5.2
batch_size = 1 # taken from pix2pix paper 5.2

# A = np.load('dataset_y.npy') # original images
# B = np.load('dataset_x.npy') # saliency
# A = np.load('val_dataset_y.npy') # original images
# B = np.load('val_dataset_x.npy') # saliency
A = np.load('dataset_y_1.npy') # original images
B = np.load('dataset_x_1.npy') # saliency
# A = np.load('val_dataset_y_1.npy') # original images
# B = np.load('val_dataset_x_1.npy') # saliency

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
        if step % 10000 == 0:
        # if step % 5000 == 0:
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
            print("step is :", step)
            plt.show()
            # plt.savefig("images/iter_" + str(step) + ".jpg")
            # plt.savefig("images/iter_%d.jpg" % step)
            # plt.savefig('images/iter_%d.jpg' % (step+1))
            # plt.close()
        
        # Save the trained model 
        # if step % 3000 == 0:
        
        # if ((step % 20000 == 0) and (step != 0)):
        if ((step % 40000 == 0) and (step != 0)):
            # Save the model
            # save_path = saver.save(sess, "models/model.ckpt")
            # save_path = saver.save(sess, "models/model_inverse.ckpt")
            # save_path = saver.save(sess, "models/model_mse/model.ckpt")
            save_path = saver.save(sess, "models/model_l1_Adv_2/model.ckpt")
            print("Model saved in file: %s" % save_path)
'''        
 


'''
# Use the 5001-10000 training images to train the model 
from __future__ import division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from pix2pix import Pix2pix
from pix2pix_New import Pix2pix
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # for limiting the debug information of tensorflow
# import cv2

# iters = 200*400 + 2 # taken from pix2pix paper 5.2
# iters = 200*100 + 2 # taken from pix2pix paper 5.2
# iters = 200*10 + 2 # taken from pix2pix paper 5.2
# iters = 500 + 2 # taken from pix2pix paper 5.2
iters = 200*100 + 2 # taken from pix2pix paper 5.2
batch_size = 1 # taken from pix2pix paper 5.2

# A = np.load('dataset_y.npy') # original images
# B = np.load('dataset_x.npy') # saliency
# A = np.load('val_dataset_y.npy') # original images
# B = np.load('val_dataset_x.npy') # saliency
A = np.load('dataset_y_2.npy') # original images
B = np.load('dataset_x_2.npy') # saliency
# A = np.load('val_dataset_y_1.npy') # original images
# B = np.load('val_dataset_x_1.npy') # saliency

with tf.device('/gpu:0'):
    # model = Pix2pix(256, 256, ichan=3, ochan=3)
    # model = Pix2pix(128, 128, ichan=3, ochan=3)
    model = Pix2pix(240, 320, ichan=3, ochan=3)

saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/models/model_l1_Adv_2/model.ckpt")
    print("Model Restored Successfully !!!")    
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
        if step % 10000 == 0:
        # if step % 5000 == 0:
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
            print("step is :", step)
            plt.show()
            # plt.savefig("images/iter_" + str(step) + ".jpg")
            # plt.savefig("images/iter_%d.jpg" % step)
            # plt.savefig('images/iter_%d.jpg' % (step+1))
            # plt.close()
        
        # Save the trained model 
        # if step % 3000 == 0:
        
        # if ((step % 20000 == 0) and (step != 0)):
        if ((step % 20000 == 0) and (step != 0)):
            # Save the model
            # save_path = saver.save(sess, "models/model.ckpt")
            # save_path = saver.save(sess, "models/model_inverse.ckpt")
            # save_path = saver.save(sess, "models/model_mse/model.ckpt")
            save_path = saver.save(sess, "models/model_l1_Adv_3/model.ckpt")
            print("Model saved in file: %s" % save_path)
''' 


