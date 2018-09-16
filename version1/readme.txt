Version of the CFS-GAN Saliency Prediction Method

Some precious experience about Debugging Tensorflow:
Refer to :example_New.py, pix2pix_New.py about new loss (KL especially), there are some debugging techniques

Tensorflow is hard to call independent Numpy functions/operations, so we would better to establish our own loss using tf.XXXX operations (this can make sure that the operations can be involved into the static graph of tensorflow)
