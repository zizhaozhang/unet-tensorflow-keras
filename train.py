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

## TODO: Writing your own data layer to read image and corresponding groundtruth mask
## see below for detailed output format
from loader import DataLayer

# configuration session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
K.set_learning_phase(0)

# define opts
opt  = {
        'batch_size': 2,
        'learning_rate': 0.0001,
        'lr_decay': 0.5,
        'save_model_every': 100,
        'checkpoint_path': 'checkpoints/unet',
        'epoch': 50,
        'load_from_checkpoint': 'unet-654'
}

# define data loader
loader = MuscleDataLayer(opt)
iter_epoch = loader.get_iter_epoch()
# define model, the last dimension is the channel
img_shape = (None, 300, 300, 3)
img = tf.placeholder(tf.float32, shape=img_shape)
label = tf.placeholder(tf.float32, shape=(None, 300, 300, 1))
with tf.name_scope('unet'):
    pred = UNet().create_model(img_shape, backend='tf', tf_input=img)
# define loss
with tf.name_scope('cross_entropy'):
    cross_entropy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=pred))
# define optimzer
global_step = tf.Variable(0, name='global_step', trainable=False)
with tf.name_scope('learning_rate'):
    learning_rate = tf.train.exponential_decay(0.0001, global_step,
                                           iter_epoch*3, 0.5, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss, global_step=global_step)
# compute dice score for simple evaluation during training
with tf.name_scope('dice_eval'):
    dice_evaluator = tf.reduce_mean(dice_coef(label, pred))

# visualization conv1
conv1_kernel = tf.global_variables()[0] # get the first convolutional kernel
# normalize to the rgb space
x_min = tf.reduce_min(conv1_kernel)
x_max = tf.reduce_max(conv1_kernel)
kernel_0_to_1 = (conv1_kernel - x_min) / (x_max - x_min)
kernel_transposed = tf.transpose(kernel_0_to_1, [3, 0, 1, 2]) # transpose

# deine summary for tensorboard
tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
tf.summary.image('prediction', pred)
tf.summary.scalar('learning_rate', learning_rate)
tf.summary.scalar('dice_score', dice_evaluator)
tf.summary.image('conv1/kernel', kernel_transposed)
summary_merged = tf.summary.merge_all()
# define saver
if os.path.isdir('./train_log'): os.system('rm train_log/*')
train_writer = tf.summary.FileWriter('./train_log', sess.graph)
saver = tf.train.Saver()

def train():
    tot_iter = iter_epoch * opt['epoch']

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    with tf.Graph().as_default(), sess.as_default():
       
        # restore from a checkpoint if exists
        # the name_scope can not change 
        try:
            saver.restore(sess, os.path.join('checkpoints',opt['load_from_checkpoint']))
            print ('--> load from checkpoint '+opt['load_from_checkpoint'])
        except Exception, e:
                print ('unable to load checkpoint ...' + str(e))
        
        start = global_step.eval()
        for it in range(start, tot_iter):
            x_batch, y_batch = loader.load_batch()
            # tensorflow want a different tensor order
            feed_dict = {   img: x_batch.transpose((0,2,3,1)),
                            label:y_batch.transpose((0,2,3,1))}
            _, loss, summary, dice_score = sess.run([ train_step, 
                                                cross_entropy_loss, 
                                                summary_merged, 
                                                dice_evaluator], \
                                        feed_dict=feed_dict)
            global_step.assign(it).eval()
            train_writer.add_summary(summary, it)
            if it % 10 == 0 : 
                print ('epoch %f: loss=%f, dice_score=%f' % (float(it)/iter_epoch, loss, dice_score))
            if it % iter_epoch == 0:
                saver.save(sess, opt['checkpoint_path'], global_step=global_step)
                print ('save a checkpoint at '+ opt['checkpoint_path']+'-'+str(it))
        
def main(_):
    train()

if __name__ == "__main__":
    tf.app.run()