'''
 * @author [Zizhao Zhang]
 * @email [zizhao@cise.ufl.edu]
 * @create date 2017-05-25 02:20:01
 * @modify date 2017-05-25 02:20:01
 * @desc [description]
'''

import argparse
import os
parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=5, help='input batch size')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=0.9, help='learning rate decay')
parser.add_argument('--epoch', type=int, default=50, help='# of epochs')
parser.add_argument('--imSize', type=int, default=256, help='then crop to this size')
parser.add_argument('--iter_epoch', type=int, default=0, help='# of iteration as an epoch')
parser.add_argument('--num_class', type=int, default=2, help='# of classes')
parser.add_argument('--checkpoint_path', type=str, default='', help='where checkpoint saved')
parser.add_argument('--data_path', type=str, default='', help='where dataset saved. See loader.py to know how to organize the dataset folder')
parser.add_argument('--load_from_checkpoint', type=str, default='', help='where checkpoint saved')

opt = parser.parse_args()

args = vars(opt)
print('------------ Options -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

if opt.checkpoint_path != '' and not os.path.isdir(opt.checkpoint_path):
    os.mkdir(opt.checkpoint_path)

# hardcode here
dataset_mean = [0.5,0.5,0.5]
dataset_std = [0.5,0.5,0.5]