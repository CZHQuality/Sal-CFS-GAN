'''
# The original loss
import numpy as np
import tensorflow as tf
from model.discriminator import Discriminator
# from model.generator import Generator
from model.generator_New import Generator

class Pix2pix(object):
    def __init__(self, width, height, ichan, ochan, l1_weight=100., lr=0.0002, beta1=0.5):
    # def __init__(self, width, height, ichan, ochan, l1_weight=1000., lr=0.0002, beta1=0.5): # I enlarge the l1_weight to see what will happen
        """
            width: image width in pixel.
            height: image height in pixel.
            ichan: number of channels used by input images.
            ochan: number of channels used by output images.
            l1_weight: L1 loss weight.
            lr: learning rate for ADAM optimizer.
            beta1: beta1 parameter for ADAM optimizer.
        """
        self._is_training = tf.placeholder(tf.bool)

        self._g_inputs = tf.placeholder(tf.float32, [None, width, height, ichan])
        self._d_inputs_a = tf.placeholder(tf.float32, [None, width, height, ichan])
        self._d_inputs_b = tf.placeholder(tf.float32, [None, width, height, ochan])
        self._g = Generator(self._g_inputs, self._is_training, ochan)
        self._real_d = Discriminator(tf.concat([self._d_inputs_a, self._d_inputs_b], axis=3), self._is_training)
        # self._fake_d = Discriminator(tf.concat([self._d_inputs_a, self._g._decoder['cl9']['fmap']], axis=3), self._is_training, reuse=True)
        self._fake_d = Discriminator(tf.concat([self._d_inputs_a, self._g._resout['rsl7']['fmap']], axis=3), self._is_training, reuse=True)


        #self._g_loss = -tf.reduce_mean(tf.log(self._fake_d._discriminator['l5']['fmap'])) + l1_weight * tf.reduce_mean(tf.abs(self._d_inputs_b - self._g._decoder['cl9']['fmap']))
        self._g_loss = -tf.reduce_mean(tf.log(self._fake_d._discriminator['l5']['fmap'])) + l1_weight * tf.reduce_mean(tf.abs(self._d_inputs_b - self._g._resout['rsl7']['fmap']))
        self._d_loss = -tf.reduce_mean(tf.log(self._real_d._discriminator['l5']['fmap']) + tf.log(1.0 - self._fake_d._discriminator['l5']['fmap']))
        # self._g_loss = tf.reduce_mean(tf.square(self._d_inputs_b - self._g._decoder['cl9']['fmap'])) # set g_loss = mse loss !!!!!
        # self._d_loss = -tf.reduce_mean(tf.log(self._real_d._discriminator['l5']['fmap']) + tf.log(1.0 - self._fake_d._discriminator['l5']['fmap']))

        g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='G')
        with tf.control_dependencies(g_update_ops):
            self._g_train_step = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(self._g_loss,
                    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G'))

        d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='D')
        with tf.control_dependencies(d_update_ops):
            self._d_train_step = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(self._d_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D'))

    def train_step(self, sess, g_inputs, d_inputs_a, d_inputs_b, is_training=True):
        _, dloss_curr = sess.run([self._d_train_step, self._d_loss],
            feed_dict={self._d_inputs_a : d_inputs_a, self._d_inputs_b : d_inputs_b, self._g_inputs : g_inputs, self._is_training : is_training})
        _, gloss_curr = sess.run([self._g_train_step, self._g_loss],
                feed_dict={self._g_inputs : g_inputs, self._d_inputs_a : d_inputs_a,   self._d_inputs_b : d_inputs_b,self._is_training : is_training})
        return (gloss_curr, dloss_curr)

    # def sample_generator(self, sess, g_inputs, is_training=False):
      #   return sess.run(self._g._decoder['cl9']['fmap'], feed_dict={self._g_inputs : g_inputs, self._is_training : is_training})

    def sample_generator(self, sess, g_inputs, is_training=False):
        return sess.run(self._g._resout['rsl7']['fmap'], feed_dict={self._g_inputs : g_inputs, self._is_training : is_training})
'''


# We add the pixel-wise losses (L1, KLD, NSS and CC loss) in this part, and we also add the Histogram Loss in this code
import numpy as np
import tensorflow as tf
from model.discriminator import Discriminator
# from model.generator import Generator # Traditional U-Net, for abalation comparison
from model.generator_New import Generator # U-Net + ResBlock + HistLoss, for abalation comparison
# from model.generator_New_2 import Generator # U-Net + ResBlock + Center-Surround Connection + HistLoss, for abalation comparison
# import New_Loss

class Pix2pix(object):
    def __init__(self, width, height, ichan, ochan, l1_weight=100., lr=0.0002, beta1=0.5):
    # def __init__(self, width, height, ichan, ochan, l1_weight=1000., lr=0.0002, beta1=0.5): # I enlarge the l1_weight to see what will happen
        """
            width: image width in pixel.
            height: image height in pixel.
            ichan: number of channels used by input images.
            ochan: number of channels used by output images.
            l1_weight: L1 loss weight.
            lr: learning rate for ADAM optimizer.
            beta1: beta1 parameter for ADAM optimizer.
        """
        self._is_training = tf.placeholder(tf.bool)

        self._g_inputs = tf.placeholder(tf.float32, [None, width, height, ichan])
        self._d_inputs_a = tf.placeholder(tf.float32, [None, width, height, ichan])
        self._d_inputs_b = tf.placeholder(tf.float32, [None, width, height, ochan])
        self._d_inputs_c = tf.placeholder(tf.float32, [None, width, height, ochan])
        self._g = Generator(self._g_inputs, self._is_training, ochan)
        self._real_d = Discriminator(tf.concat([self._d_inputs_a, self._d_inputs_b], axis=3), self._is_training)
        # self._fake_d = Discriminator(tf.concat([self._d_inputs_a, self._g._decoder['cl9']['fmap']], axis=3), self._is_training, reuse=True)
        self._fake_d = Discriminator(tf.concat([self._d_inputs_a, self._g._resout['rsl7']['fmap']], axis=3), self._is_training, reuse=True)


        #self._g_loss = -tf.reduce_mean(tf.log(self._fake_d._discriminator['l5']['fmap'])) + l1_weight * tf.reduce_mean(tf.abs(self._d_inputs_b - self._g._decoder['cl9']['fmap']))
        # self._g_loss = -tf.reduce_mean(tf.log(self._fake_d._discriminator['l5']['fmap'])) + l1_weight * tf.reduce_mean(tf.abs(self._d_inputs_b - self._g._resout['rsl7']['fmap']))
        # with tf.Session() as sess:
          #  img_true = (self._d_inputs_b).eval()
          #  img_pred = (self._g._resout['rsl7']['fmap']).eval()
        # img_true = (self._d_inputs_b)
        # sess.run(
        # img_true = (self._g._resout['rsl7']['fmap'])
        # img_pred = (self._g._resout['rsl7']['fmap'])
        # print("img_true shape:", img_true.shape)
        # Saliency_Loss = New_Loss.final_loss(img_true, img_pred)
        
        
        # img_true_tensor = tf.to_float(self._d_inputs_b[0,:,:,1], name='ToFloat')
        img_true_tensor = tf.to_float(self._d_inputs_b, name='ToFloat')
        img_true_tensor_pts = tf.to_float(self._d_inputs_c, name='ToFloat')
        # img_true_tensor = tf.to_double(self._d_inputs_b, name='ToDouble')
        # img_pred_tensor = tf.to_float(self._g._resout['rsl7']['fmap'][0,:,:,1], name='ToFloat')
        # img_pred_tensor = tf.to_float(self._d_inputs_b, name='ToFloat') # In this case, KLD ~= 0, and CC ~= 1, and Hist Chi-Square Score ~= 0, just for testing the correctness of metric 
        img_pred_tensor = tf.to_float(self._g._resout['rsl7']['fmap'], name='ToFloat') # real predicted saliency map
        img_pred_tensor = img_pred_tensor + 1 # normalize because generator use tanh activation in its output layer
        
        # map2_NSS = img_pred_tensor

        # sum_true = tf.constant(tf.reduce_sum(img_true_tensor))
        # sum_pred = tf.constant(tf.reduce_sum(img_pred_tensor))
        big1 = tf.reduce_max(img_true_tensor)
        sml1 = tf.reduce_min(img_true_tensor)
        img_true_tensor = (img_true_tensor - sml1) / (big1 - sml1) # min-max normalization
        big2 = tf.reduce_max(img_pred_tensor)
        sml2 = tf.reduce_min(img_pred_tensor)
        img_pred_tensor = (img_pred_tensor - sml2) / (big2 - sml2)
        big3 = tf.reduce_max(img_true_tensor_pts)
        sml3 = tf.reduce_min(img_true_tensor_pts)
        img_true_tensor_pts = (img_true_tensor_pts - sml3) / (big3 - sml3)

        map1_CC = img_true_tensor
        map2_CC = img_pred_tensor
        map1_NSS = img_true_tensor_pts
        map2_NSS = img_pred_tensor
        shan = img_pred_tensor
        

        # print("img_true_tensor is :::::::", img_true_tensor)
        sum_true = tf.reduce_sum(img_true_tensor) 
        # print("sum_true is ::::::::::", sum_true)
        sum_pred = tf.reduce_sum(img_pred_tensor)
        map2 = img_true_tensor / sum_true # make sure the elements of map2/map1 are all in range [0, 1]
        map1 = img_pred_tensor / sum_pred
        # epsilon = tf.constant(1e-08)
        epsilon = 1e-08
        epsilon = tf.to_float(epsilon, name='ToFloat')

        KLD_1 = tf.divide(map2, tf.add(map1, epsilon, name=None), name=None)
        KLD_2 = tf.log(tf.add(epsilon, KLD_1, name=None), name=None)
        KLD_3 = tf.reduce_sum(tf.multiply(map2, KLD_2, name=None))

 
        # img_true_tensor = tf.to_float(self._d_inputs_b, name='ToFloat')
        # img_pred_tensor = tf.to_float(self._g._resout['rsl7']['fmap'], name='ToFloat')
        # img_pred_tensor = img_pred_tensor + 1 # normalize because generator use tanh activation in its output layer
        # map1_CC = (map1_CC - tf.reduce_mean(map1_CC)) / 
        mean_1,var_1 = tf.nn.moments(map1_CC, axes = [0,1,2,3])
        mean_2,var_2 = tf.nn.moments(map2_CC, axes = [0,1,2,3])
        map1_CC = (map1_CC - mean_1) / tf.sqrt( tf.add(var_1, epsilon)) # z-score normalization
        map2_CC = (map2_CC - mean_2) / tf.sqrt( tf.add(var_2, epsilon))

        CC_1 = tf.multiply( (map1_CC - mean_1), (map2_CC - mean_2) )
        CC_1 = tf.reduce_sum(CC_1)
        CC_2 = tf.reduce_sum(tf.square((map1_CC - mean_1))) * tf.reduce_sum(tf.square((map2_CC - mean_2)))
        CC_2 = tf.sqrt(CC_2) + epsilon
        CC_3 = CC_1 / CC_2


        mean_3, var_3 = tf.nn.moments(map2_NSS, axes = [0,1,2,3])
        # big4 = tf.reduce_max(map1_NSS)
        map2_NSS = (map2_NSS - mean_3) / tf.sqrt( tf.add(var_3, epsilon) )
        # NSS_1 = tf.multiply(map1_NSS, map2_NSS)
        # NSS_2 = tf.reduce_mean(NSS_1)
        NSS_idx = tf.where(map1_NSS > 0.1) # should not be 0, because there are a lot of 0.00XXX in map1_NSS due to float format
        NSS_1 = tf.gather_nd(map2_NSS, NSS_idx)
        # dims = NSS_1.shape
        NSS_2 = tf.reduce_mean(NSS_1)
        # NSS_3 = tf.gather_nd(map1_NSS, NSS_idx)

        HistCS_img_GT = img_true_tensor
        HistCS_img_GT = tf.multiply(HistCS_img_GT, 255.0) 
        # HistCS_img_GT = tf.convert_to_tensor(HistCS_img_GT)
        HistCS_img_GT = HistCS_img_GT[0, :, :, 1]

        HistCS_img_SM = img_pred_tensor
        HistCS_img_SM = tf.multiply(HistCS_img_SM, 255.0)
        HistCS_img_SM = HistCS_img_SM[0, :, :, 1]
        # HistCS_img_SM = tf.convert_to_tensor(HistCS_img_SM)
        # HistCS_img_GT = img_true_tensor
        # HistCS_img_SM = img_pred_tensor
        # HistCS_img_GT = tf.cast(HistCS_img_GT, tf.float32)
        HistCS_img_GT = tf.to_float(HistCS_img_GT, name='ToFloat')
        sp_1 = tf.shape(HistCS_img_GT)
        # HistCS_img_SM = tf.cast(HistCS_img_SM, tf.float32)
        HistCS_img_SM = tf.to_float(HistCS_img_SM, name='ToFloat')
        dada = tf.reduce_max(HistCS_img_GT)
        xiaoxiao = tf.reduce_min(HistCS_img_GT)
        dada2 = tf.reduce_max(HistCS_img_SM)
        xiaoxiao2 = tf.reduce_min(HistCS_img_SM)
        
        nbins = 256
        VALUE_RANGE = [0, 255]
        VALUE_RANGE = tf.to_float(VALUE_RANGE, name='ToFloat')

        # sp_1 = tf.shape(HistCS_img_GT)
        # sp_2 = tf.shape(HistCS_img_SM)
        # sp_3 = tf.shape(img_true_tensor)
        
        
        hist_GT = tf.histogram_fixed_width(HistCS_img_GT, VALUE_RANGE, nbins)
        hist_GT = tf.to_float(hist_GT, name='ToFloat')
        da_1 = tf.reduce_max(hist_GT) # MINMAX normalization for Histogram
        da_1 = tf.to_float(da_1, name='ToFloat')
        xiao_1 = tf.reduce_min(hist_GT)
        xiao_1 = tf.to_float(xiao_1, name='ToFloat')
        hist_GT = tf.divide(tf.subtract(hist_GT, xiao_1), (da_1 - xiao_1))
        hist_GT = tf.add(tf.multiply(hist_GT, 255), 0)

        hist_SM = tf.histogram_fixed_width(HistCS_img_SM, VALUE_RANGE, nbins)
        hist_SM = tf.to_float(hist_SM, name='ToFloat')
        da_2 = tf.reduce_max(hist_SM) # MINMAX normalization for Histogram
        da_2 = tf.to_float(da_2, name='ToFloat')
        xiao_2 = tf.reduce_min(hist_SM)
        xiao_2 = tf.to_float(xiao_2, name='ToFloat')
        hist_SM = tf.divide(tf.subtract(hist_SM, xiao_2), (da_2 - xiao_2))
        hist_SM = tf.add(tf.multiply(hist_SM, 255), 0)

        score_hist_1 = tf.square(tf.subtract(hist_GT, hist_SM)) 
        score_hist_2 = tf.add(hist_GT, hist_SM)
        score_hist_1 = tf.to_float(score_hist_1, name='ToFloat')
        score_hist_2 = tf.to_float(score_hist_2, name='ToFloat')
        score_hist_2 = tf.add(score_hist_2, epsilon) # plus epsilon to prevent denominator is 0
        score_hist_3 = tf.reduce_sum(tf.divide(score_hist_1, score_hist_2)) * 2
        score_hist_3_final = score_hist_3 / 1000
        


        # print("KLD is :", KLD_3)

        # self.KLD = sum_true # This is just a monitor output, for debugging, for output the variables you want to see, refer to "example_New.py" line 47
        self.monitor = [KLD_3, CC_3, NSS_2, score_hist_3_final]  # This is just a monitor output, for debugging, for output the variables you want to see, refer to "example_New.py" line 47
        # self.monitor = [dada, xiaoxiao, dada2, xiaoxiao2, sp_1, score_hist_3]  # This is just a monitor output, for debugging, for output the variables you want to see, refer to "example_New.py" line 47
        # And pay attention that: we have to add "self." in front of "monitor", so that we can calculate it in sess.run (refer to the 172 line of this code)
        
        l1_weight = 10 # !!!!! Attention: for L1 loss, default value is 100, the smaller the better. In fine-tune stage, we use the smaller l1_weight to emphasis the saliency-specific losses
        l2_weight = 100 # for KLD, the smaller the better
        l3_weight = -20 # for CC, the bigger the better
        # l4_weight = -10 # for NSS, the bigger the better
        l4_weight = -20 # default: -20  for NSS, the bigger the better, we emphsis the weight of NSS and compare the performance
        l5_weight = 10 # default:10 for Histogram Loss (Alternative Chi-Square Hist Loss), the smaller the better

        # self._g_loss = -tf.reduce_mean(tf.log(self._fake_d._discriminator['l5']['fmap'])) + l1_weight * tf.reduce_mean(tf.abs(self._d_inputs_b - self._g._resout['rsl7']['fmap']))
        # self._g_loss = -tf.reduce_mean(tf.log(self._fake_d._discriminator['l5']['fmap'])) + l1_weight * tf.reduce_mean(tf.abs(self._d_inputs_b - self._g._resout['rsl7']['fmap'])) + l2_weight * KLD_3
        # self._g_loss = -tf.reduce_mean(tf.log(self._fake_d._discriminator['l5']['fmap'])) + \
          #              l1_weight * tf.reduce_mean(tf.abs(self._d_inputs_b - self._g._resout['rsl7']['fmap'])) + l2_weight * KLD_3 + l3_weight * CC_3
        # self._g_loss = -tf.reduce_mean(tf.log(self._fake_d._discriminator['l5']['fmap'])) + \
          #              l1_weight * tf.reduce_mean(tf.abs(self._d_inputs_b - self._g._resout['rsl7']['fmap'])) + l2_weight * KLD_3 + l3_weight * CC_3 + l4_weight * NSS_2
        
        self._g_loss = -tf.reduce_mean(tf.log(self._fake_d._discriminator['l5']['fmap'])) + \
                        l1_weight * tf.reduce_mean(tf.abs(self._d_inputs_b - self._g._resout['rsl7']['fmap'])) + l2_weight * KLD_3 + l3_weight * CC_3 + l4_weight * NSS_2 + l5_weight * score_hist_3_final
        
        # self._g_loss = l2_weight * KLD_3
        # self._g_loss = l3_weight * CC_3

        self._d_loss = -tf.reduce_mean(tf.log(self._real_d._discriminator['l5']['fmap']) + tf.log(1.0 - self._fake_d._discriminator['l5']['fmap']))
        # self._g_loss = tf.reduce_mean(tf.square(self._d_inputs_b - self._g._decoder['cl9']['fmap'])) # set g_loss = mse loss !!!!!
        # self._d_loss = -tf.reduce_mean(tf.log(self._real_d._discriminator['l5']['fmap']) + tf.log(1.0 - self._fake_d._discriminator['l5']['fmap']))

        g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='G')
        with tf.control_dependencies(g_update_ops):
            self._g_train_step = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(self._g_loss,
                    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G'))

        d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='D')
        with tf.control_dependencies(d_update_ops):
            self._d_train_step = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(self._d_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D'))

    
    # def train_step(self, sess, g_inputs, d_inputs_a, d_inputs_b, is_training=True):
    def train_step(self, sess, g_inputs, d_inputs_a, d_inputs_b, d_inputs_c, is_training=True):
        _, dloss_curr = sess.run([self._d_train_step, self._d_loss],
            feed_dict={self._d_inputs_a : d_inputs_a, self._d_inputs_b : d_inputs_b, self._g_inputs : g_inputs, self._is_training : is_training})

        # _, gloss_curr = sess.run([self._g_train_step, self._g_loss],
          #      feed_dict={self._g_inputs : g_inputs, self._d_inputs_a : d_inputs_a,   self._d_inputs_b : d_inputs_b,self._is_training : is_training})        
        # _, gloss_curr, _monitor = sess.run([self._g_train_step, self._g_loss, self.monitor],
          #      feed_dict={self._g_inputs : g_inputs, self._d_inputs_a : d_inputs_a,   self._d_inputs_b : d_inputs_b,self._is_training : is_training})
        _, gloss_curr, _monitor = sess.run([self._g_train_step, self._g_loss, self.monitor],
                feed_dict={self._g_inputs : g_inputs, self._d_inputs_a : d_inputs_a,  self._d_inputs_b : d_inputs_b, self._d_inputs_c : d_inputs_c, self._is_training : is_training})
        return (gloss_curr, dloss_curr, _monitor)

    # def sample_generator(self, sess, g_inputs, is_training=False):
      #   return sess.run(self._g._decoder['cl9']['fmap'], feed_dict={self._g_inputs : g_inputs, self._is_training : is_training})

    def sample_generator(self, sess, g_inputs, is_training=False):
        return sess.run(self._g._resout['rsl7']['fmap'], feed_dict={self._g_inputs : g_inputs, self._is_training : is_training})




