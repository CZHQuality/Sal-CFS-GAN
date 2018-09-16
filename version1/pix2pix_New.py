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


# We add the KLD, NSS and CC loss in this part
import numpy as np
import tensorflow as tf
from model.discriminator import Discriminator
# from model.generator import Generator
from model.generator_New import Generator
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
        # img_true_tensor = tf.to_double(self._d_inputs_b, name='ToDouble')
        # img_pred_tensor = tf.to_float(self._g._resout['rsl7']['fmap'][0,:,:,1], name='ToFloat')
        img_pred_tensor = tf.to_float(self._g._resout['rsl7']['fmap'], name='ToFloat')
        img_pred_tensor = img_pred_tensor + 1 # normalize because generator use tanh activation in its output layer
        # img_pred_tensor = tf.to_float(self._d_inputs_b, name='ToFloat') # In this case, KLD ~= 0

        # sum_true = tf.constant(tf.reduce_sum(img_true_tensor))
        # sum_pred = tf.constant(tf.reduce_sum(img_pred_tensor))
        big1 = tf.reduce_max(img_true_tensor)
        sml1 = tf.reduce_min(img_true_tensor)
        img_true_tensor = (img_true_tensor - sml1) / (big1 - sml1)
        big2 = tf.reduce_max(img_pred_tensor)
        sml2 = tf.reduce_min(img_pred_tensor)
        img_pred_tensor = (img_pred_tensor - sml2) / (big2 - sml2)

        # print("img_true_tensor is :::::::", img_true_tensor)
        sum_true = tf.reduce_sum(img_true_tensor)
        # print("sum_true is ::::::::::", sum_true)
        sum_pred = tf.reduce_sum(img_pred_tensor)

        img_true_tensor = img_true_tensor / sum_true
        img_pred_tensor = img_pred_tensor / sum_pred
        # epsilon = tf.constant(1e-08)
        epsilon = 1e-08
        epsilon = tf.to_float(epsilon, name='ToFloat')
        map2 = img_true_tensor
        map1 = img_pred_tensor

        KLD_1 = tf.divide(map2, tf.add(map1, epsilon, name=None), name=None)
        KLD_2 = tf.log(tf.add(epsilon, KLD_1, name=None), name=None)
        KLD_3 = tf.reduce_sum(tf.multiply(map2, KLD_2, name=None))
        # print("KLD is :", KLD_3)

        # self.KLD = sum_true # This is just a monitor output, for debugging, for output the variables you want to see, refer to "example_New.py" line 47
        self.monitor = sum_true # This is just a monitor output, for debugging, for output the variables you want to see, refer to "example_New.py" line 47
        # And pay attention that: we have to add "self." in front of "monitor", so that we can calculate it in sess.run (refer to the 172 line of this code)

        l2_weight = 100
        # self._g_loss = -tf.reduce_mean(tf.log(self._fake_d._discriminator['l5']['fmap'])) + l1_weight * tf.reduce_mean(tf.abs(self._d_inputs_b - self._g._resout['rsl7']['fmap']))
        self._g_loss = -tf.reduce_mean(tf.log(self._fake_d._discriminator['l5']['fmap'])) + l1_weight * tf.reduce_mean(tf.abs(self._d_inputs_b - self._g._resout['rsl7']['fmap'])) + l2_weight * KLD_3

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

        # _, gloss_curr = sess.run([self._g_train_step, self._g_loss],
          #      feed_dict={self._g_inputs : g_inputs, self._d_inputs_a : d_inputs_a,   self._d_inputs_b : d_inputs_b,self._is_training : is_training})        
        _, gloss_curr, _monitor = sess.run([self._g_train_step, self._g_loss, self.monitor],
                feed_dict={self._g_inputs : g_inputs, self._d_inputs_a : d_inputs_a,   self._d_inputs_b : d_inputs_b,self._is_training : is_training})
        return (gloss_curr, dloss_curr, _monitor)

    # def sample_generator(self, sess, g_inputs, is_training=False):
      #   return sess.run(self._g._decoder['cl9']['fmap'], feed_dict={self._g_inputs : g_inputs, self._is_training : is_training})

    def sample_generator(self, sess, g_inputs, is_training=False):
        return sess.run(self._g._resout['rsl7']['fmap'], feed_dict={self._g_inputs : g_inputs, self._is_training : is_training})




