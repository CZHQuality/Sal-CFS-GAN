import tensorflow as tf 
import theano.tensor as T
import keras.backend as K
import cv2
import numpy as np 


shape_r = 240
# number of cols of input images
shape_c = 320
# number of rows of downsampled maps
shape_r_out = 240
# number of cols of model outputs
shape_c_out = 320
# final upsampling factor


# KL-Divergence Loss
def kl_divergence(y_true, y_pred):
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)), 
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=2), axis=2)), 
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=2), axis=2)), 
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    return 10 * K.sum(K.sum(y_true * K.log((y_true / (y_pred + K.epsilon())) + K.epsilon()), axis=-1), axis=-1) # weight: 10


# Correlation Coefficient Loss
def correlation_coefficient(y_true, y_pred):
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)), 
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=2), axis=2)), 
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=2), axis=2)), 
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)

    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    N = shape_r_out * shape_c_out
    sum_prod = K.sum(K.sum(y_true * y_pred, axis=2), axis=2)
    sum_x = K.sum(K.sum(y_true, axis=2), axis=2)
    sum_y = K.sum(K.sum(y_pred, axis=2), axis=2)
    sum_x_square = K.sum(K.sum(K.square(y_true), axis=2), axis=2)
    sum_y_square = K.sum(K.sum(K.square(y_pred), axis=2), axis=2)

    num = sum_prod - ((sum_x * sum_y) / N)
    den = K.sqrt((sum_x_square - K.square(sum_x) / N) * (sum_y_square - K.square(sum_y) / N))

    # print("num is :", num)
    # print("den is :", den)
    return -2 * num / den # weight: -2



# Normalized Scanpath Saliency Loss
def nss(y_true, y_pred):
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)), 
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    y_pred /= max_y_pred
    y_pred_flatten = K.batch_flatten(y_pred)

    y_mean = K.mean(y_pred_flatten, axis=-1)
    y_mean = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.expand_dims(y_mean)), 
                                                               shape_r_out, axis=-1)), shape_c_out, axis=-1)

    y_std = K.std(y_pred_flatten, axis=-1)
    y_std = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.expand_dims(y_std)), 
                                                              shape_r_out, axis=-1)), shape_c_out, axis=-1)

    y_pred = (y_pred - y_mean) / (y_std + K.epsilon())

    return -(K.sum(K.sum(y_true * y_pred, axis=2), axis=2) / K.sum(K.sum(y_true, axis=2), axis=2)) # weight: -1


def final_loss(y_true, y_pred):


    sess = tf.InteractiveSession()
    y_true = y_true.eval(session = sess)
    y_pred = y_pred.eval(session = sess)
    # y_true = sess.run(y_true)

    a_1 = np.expand_dims(y_true, axis=0) # add the batch_size to channel 0
    print("a_1 shape :", a_1.shape)
    print("a_1 is :", a_1)
    a_2 = a_1[:,:,:,1] # extract the one of the color channel from 3 channels: one of R/G/B channel
    a_3 = np.zeros((1, 1, 240, 320))
    a_3[0, 0] = a_2.astype(np.float32) # subset of a_3, the inner element of a_3
    a_3[0, 0] /= 255.0
    
    b_1 = np.expand_dims(y_pred, axis=0) # add the batch_size to channel 0
    b_2 = b_1[:,:,:,1] # extract the one of the color channel from 3 channels: one of R/G/B channel
    b_3 = np.zeros((1, 1, 240, 320))
    b_3[0, 0] = b_2.astype(np.float32) # subset of b_3, the inner element of b_3
    b_3[0, 0] /= 255.0
    

    '''
    y_pred = cv2.imread('/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/SALICON_300_val/0_SM.jpg', 0)
    b_3 = np.zeros((1, 1, 240, 320))
    b_3[0, 0] = y_pred.astype(np.float32)
    b_3[0, 0] /= 255.0
    '''


    kld = kl_divergence(a_3, b_3)
    cc = correlation_coefficient(a_3, b_3)
    nss_score = nss(a_3, b_3)
    final_loss = kld.eval()[0][0] + cc.eval()[0][0] + nss_score.eval()[0][0]
    
    '''
    print("KLD is :", kld.eval()[0][0]/(10))
    print("CC is :", cc.eval()[0][0]/(-2))
    print("NSS is :", nss_score.eval()[0][0]/(-1))  
    print("Final Loss is :", final_loss)
    '''

    return final_loss


if __name__ == '__main__': 
    img_true = np.load('val_dataset_x.npy')
    i = 0
    img_true = img_true[i]
    img_pred = np.load('val_dataset_x.npy')
    img_pred = img_pred[i]

    f_loss = final_loss(img_true, img_pred)




'''
if __name__ == '__main__': 
    img_true = cv2.imread('/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/SALICON_300_val/1_GT.jpg', 0) # 0: read the GrayScale image format
    img_pred = cv2.imread('/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/SALICON_300_val/1_SM.jpg', 0)


    ims = np.zeros((1, 1, 240, 320))
    ims[0, 0] = img_true.astype(np.float32)
    ims[0, 0] /= 255.0

    ims2 = np.zeros((1, 1, 240, 320))
    ims2[0, 0] = img_pred.astype(np.float32)
    ims2[0, 0] /= 255.0

    [height, width] = img_pred.shape 
    print("HWC is :", height, width)
    kld = kl_divergence(ims, ims2)
    # print("KLD is :", kld.eval())
    print("KLD is :", kld.eval()[0][0]/(10))

    cc = correlation_coefficient(ims, ims2)
    # print("CC is :", cc.eval())
    print("CC is :", cc.eval()[0][0]/(-2))

    nss = nss(ims, ims2)
    # print("NSS is :", nss.eval())   
    print("NSS is :", nss.eval()[0][0]/(-1))  


    final_loss = kld.eval()[0][0] + cc.eval()[0][0] + nss.eval()[0][0]
    print("Final Loss is :", final_loss)
'''



'''
if __name__ == '__main__': 
    # img_true = cv2.imread('/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/SALICON_300_val/1_GT.jpg', 0) # 0: read the GrayScale image format
    img_pred = cv2.imread('/media/ubuntu/CZHhy/GAN/BorjiGAN/Code/result/SALICON_300_val/0_SM.jpg', 0)




    # dataset_X = np.zeros((len(names), 240, 320, 3)) #!!!!!!!!!!!!! shape format in our code : batch_size, height, width, channel
    # dataset_Y = np.zeros((len(names), 240, 320, 3))

    img_true = np.load('val_dataset_x.npy') # GT of SALICON 1-300 images
    # a = np.expand_dims(img_true[step % img_true.shape[0]], axis=0)
    i = 0
    a = np.expand_dims(img_true[i], axis=0) # add the batch_size channeld to the first channel of img_true[i] (channel 0) # (1, 240, 320, 3)
    
    # a = img_true[i]
    # print("a is :", a)
    # print("shape of a:", a.shape) # (240, 320, 3)
    
    a = a[:,:,:,1] # a = a # Which one is right ??
    print("a is :", a)
    print("shape of a:", a.shape) # (1, 240, 320)
    
    # bbb = np.zeros((1, 1, 240, 320)) 
    # bbb = a.astype(np.float32)
    # print("bbb is :", bbb)
    # print("shape of bbb:", bbb.shape)
    # a - bbb

    # ccc = a.transpose((0, 3, 1, 2))
    # print("shape of ccc:", ccc.shape)
    # a - bbb
    # a - ccc
    # print("look: ", bbb - ccc)
    
    
    

    # ims[:, :, :, 0] -= 103.939
    # ims[:, :, :, 1] -= 116.779
    # ims[:, :, :, 2] -= 123.68
    # ims = ims.transpose((0, 3, 1, 2))    
    # ims = np.zeros((len(paths), 1, shape_r, shape_c))


    ims = np.zeros((1, 1, 240, 320)) # shape of SAM-ResNet/SAM-VGG tensor, batch_size, channel, height, width, should change this to unify the tensor format
    # ims[0, 0] = img_true.astype(np.float32)
    ims[0, 0] = a.astype(np.float32)
    ims[0, 0] /= 255.0
    print("shape of ims : ", ims.shape)

    ims2 = np.zeros((1, 1, 240, 320))
    ims2[0, 0] = img_pred.astype(np.float32)
    ims2[0, 0] /= 255.0

    [height, width] = img_pred.shape 
    print("HWC is :", height, width)
    kld = kl_divergence(ims, ims2)
    # print("KLD is :", kld.eval())
    print("KLD is :", kld.eval()[0][0]/(10))

    cc = correlation_coefficient(ims, ims2)
    # print("CC is :", cc.eval())
    print("CC is :", cc.eval()[0][0]/(-2))

    nss = nss(ims, ims2)
    # print("NSS is :", nss.eval())   
    print("NSS is :", nss.eval()[0][0]/(-1))  


    final_loss = kld.eval()[0][0] + cc.eval()[0][0] + nss.eval()[0][0]
    print("Final Loss is :", final_loss)
'''