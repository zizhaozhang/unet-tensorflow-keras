# Muscle Segmentation using UNet
# Modified Unet to take arbitrary input size
# Zizhao @ UF

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
import tensorflow as tf

import argparse
import os
from model import UNet
from utils import dice_coef, dice_coef_loss
from loader import MuscleDataLayer
import scipy.misc as misc
import numpy as np
import time

# configuration session
sess = tf.Session()
K.set_session(sess)

# define opts
opt  = {
        'batch_size': 2,
        'learning_rate': 0.0001,
        'lr_decay': 0.5,
        'save_model_every': 100,
        'checkpoint_path': 'checkpoints/unet',
        'epoch': 50,
        'load_from_checkpoint': 'unet-XXX'
}
# load test image
ifresize = False
inh, inw = 600, 800
test_img = misc.imread(
    os.path.join('./test_sample', '001.png'))
if ifresize:
    test_img = misc.imresize(test_img, (inh, inw))
else:
    inh, inw = test_img.shape[0], test_img.shape[1]
input_tensor = np.reshape(test_img, (1, inh, inw, 3))

# define data loader
loader = MuscleDataLayer(opt)
iter_epoch = loader.get_iter_epoch()
# define model, the last dimension is the channel

img_shape = (None, inh, inw, 3)
input_space = tf.placeholder(tf.float32, shape=img_shape)
with tf.name_scope('unet'):
    pred = UNet().create_model(img_shape, backend='tf', tf_input=input_space)

saver = tf.train.Saver()

init_op = tf.global_variables_initializer()
sess.run(init_op)

with sess.as_default():
    try:
        saver.restore(sess, os.path.join('checkpoints',opt['load_from_checkpoint']))
        print ('--> load from checkpoint '+opt['load_from_checkpoint'])
    except Exception, e:
            print ('unable to load checkpoint ...' + str(e))
    for i in range(2):
        print ('==> beginning forward ' + str(i))
        start = time.time()
        pre_mask = sess.run([pred], feed_dict={input_space: input_tensor})
        pre_mask = pre_mask[0][0][:,:,0]
        print ('time cost '+ str(time.time() - start))
        # misc.imshow(pre_mask)