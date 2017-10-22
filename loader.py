'''
 * @author [Zizhao Zhang]
 * @email [zizhao@cise.ufl.edu]
 * @create date 2017-05-19 03:06:43
 * @modify date 2017-05-19 03:06:43
 * @desc [description]
'''
# import keras
# from keras.preprocessing.image import ImageDataGenerator
# extracted the image module from kears to remove bugs of unpaired image and mask

from data_generator.image import ImageDataGenerator
import scipy.misc as misc
import numpy as np
import os, glob, itertools
from PIL import ImageFile
from PIL import Image as pil_image
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Modify this for data normalization 
def preprocess(img, label=None):
    out_img = img.copy()
    # out_img = img.copy() - 127.5
    # out_img = (img.copy() - MEAN) / STD
    if label is not None:
        if len(label.shape) == 4:
            label = label[:,:,:,0]
        out_label = label / label.max()
        return out_img, out_label.astype(np.int32)
    else:
        return out_img

def deprocess(img, label=None):
    out_img = img.copy() + 127.5
    # out_img = img.copy() * STD + MEAN
    
    if label is not None:
        # label = label / 255.0
        return out_img, label.astype(np.int32)
    else: 
        return out_img, None

def folderLoader(data_path, imSize):
    def load_img(path, grayscale=False):
        # use the same loadering funcs with keras
        img = pil_image.open(path)
        if grayscale:
            img = img.convert('L')
        if img.size != (imSize[1], imSize[0]):
            img = img.resize((imSize[1], imSize[0]))
        return np.asarray(img)

    def im_iterator(im_list, im_path, gt_path):
        for i in range(99999): #continue the iterator
             for name in im_list:
                im = load_img(os.path.join(im_path, name))
                im = im[np.newaxis,:,:,:]
                
                gt = load_img(os.path.join(gt_path, name+'.png'), grayscale=True)
                gt = gt[np.newaxis,:,:]
                im, gt = preprocess(im, gt)
                yield im, gt, name

    im_path = os.path.join(data_path, 'val/img/0/')   
    gt_path = os.path.join(data_path, 'val/gt/0/')       
    im_list = glob.glob(im_path + '*.jpg')

    im_list = [os.path.splitext(os.path.basename(a))[0] for a in im_list]
    print ('{} test images are found'.format(len(im_list))) 

    return im_iterator(im_list, im_path, gt_path), len(im_list)

'''
    Use the Keras data generators to load train and test
    Image and label are in structure:
        train/
            img/
                0/
            gt/
                0/

        test/
            img/
                0/
            gt/
                0/

'''
def dataLoader(path, batch_size, imSize, train_mode=True):

    def imerge(a, b):
        for img, label in itertools.zip_longest(a,b):
            # j is the mask: 1) gray-scale and int8
            img, label = preprocess(img, label)
            yield img, label
    
    #augmentation parms for the train generator
    if train_mode:
        train_data_gen_args = dict(
                        horizontal_flip=True,
                        vertical_flip=True,
                        )
    else:
        train_data_gen_args = dict()
        
    seed = 1
    train_image_datagen = ImageDataGenerator(**train_data_gen_args).flow_from_directory(
                                path+'img',
                                class_mode=None,
                                target_size=imSize,
                                batch_size=batch_size,
                                seed=seed,
                                shuffle=train_mode)
    train_mask_datagen = ImageDataGenerator(**train_data_gen_args).flow_from_directory(
                                path+'gt',
                                class_mode=None,
                                target_size=imSize,
                                batch_size=batch_size,
                                color_mode='grayscale',
                                seed=seed,
                                shuffle=train_mode)
                                
    samples = train_image_datagen.samples
    generator = imerge(train_image_datagen, train_mask_datagen)

    return generator, samples


  