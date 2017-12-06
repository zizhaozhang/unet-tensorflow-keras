'''
 * @author [Zizhao Zhang]
 * @email [zizhao@cise.ufl.edu]
 * @create date 2017-05-19 03:06:43
 * @modify date 2017-05-19 03:06:43
 * @desc [description]
'''

from data_generator.image import ImageDataGenerator
import scipy.misc as misc
import numpy as np
import os, glob, itertools
from PIL import ImageFile
from PIL import Image as pil_image
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Modify this for data normalization 
def preprocess(img, mean, std, label, normalize_label=True):
    out_img = img / img.max() # scale to [0,1]
    out_img = (out_img - np.array(mean).reshape(1,1,3)) / np.array(std).reshape(1,1,3) 

    if len(label.shape) == 4:
        label = label[:,:,:,0]
    if normalize_label:
        if np.unique(label).size > 2:
            print ('WRANING: the label has more than 2 classes. Set normalize_label to False')
        label = label / label.max() # if the loaded label is binary has only [0,255], then we normalize it
    return out_img, label.astype(np.int32)

def deprocess(img, mean, std, label):
    out_img = img / img.max() # scale to [0,1]
    out_img = (out_img * np.array(std).reshape(1,1,3)) + np.array(std).reshape(1,1,3) 
    out_img = out_img * 255.0

    return out_img.astype(np.uint8), label.astype(np.uint8)

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
def dataLoader(path, batch_size, imSize, train_mode=True, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]):
    # image normalization default: scale to [-1,1]
    def imerge(a, b):
        for img, label in itertools.zip_longest(a,b):
            # j is the mask: 1) gray-scale and int8
            img, label = preprocess(img, mean, std, label)
            yield img, label
    
    # augmentation parms for the train generator
    if train_mode:
        train_data_gen_args = dict(
                        horizontal_flip=True,
                        vertical_flip=True,
                        )
    else:
        train_data_gen_args = dict()
    
    # seed has to been set to synchronize img and mask generators
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


  