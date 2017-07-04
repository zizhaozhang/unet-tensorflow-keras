'''
 * @author [Zizhao Zhang]
 * @email [zizhao@cise.ufl.edu]
 * @create date 2017-05-19 03:06:43
 * @modify date 2017-05-19 03:06:43
 * @desc [description]
'''
import keras
from keras.preprocessing.image import ImageDataGenerator
import scipy.misc as misc
import numpy as np
import os, glob, itertools
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def folderLoader(data_path):
    def im_iterator(im_list, im_path, gt_path):
        for i in range(99999): #continue the iterator
            for name in im_list:
                im = misc.imread(os.path.join(im_path, name+'.jpg'))
                im = im[np.newaxis,:,:,:]
                gt = misc.imread(os.path.join(gt_path, name+'.png')).astype(np.int32)
                gt = gt[np.newaxis,:,:]
                yield im, gt, name

    im_path = os.path.join(data_path, 'test/img/0/')   
    gt_path = os.path.join(data_path, 'test/gt/0/')       
    im_list = glob.glob(im_path + '*.jpg')
    im_list = [os.path.split(a)[1][:-4] for a in im_list]
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
def dataLoader(path, batch_size, imSize):

    def imerge(a, b):
        for img, label in itertools.izip_longest(a,b):
            # j is the mask: 1) gray-scale and int8
            yield img, label[:,:,:,0].astype(np.int32)
    
    #augmentation parms for the train generator
    train_data_gen_args = dict(
                    horizontal_flip=True,
                    vertical_flip=True,
                    )
    
    seed = 1
    train_image_datagen = ImageDataGenerator(**train_data_gen_args).flow_from_directory(
                                path+'train/img',
                                class_mode=None,
                                target_size=(imSize, imSize),
                                batch_size=batch_size,
                                seed=seed)
    train_mask_datagen = ImageDataGenerator(**train_data_gen_args).flow_from_directory(
                                path+'train/gt',
                                class_mode=None,
                                target_size=(imSize, imSize),
                                batch_size=batch_size,
                                color_mode='grayscale',
                                seed=seed)

    test_image_datagen = ImageDataGenerator().flow_from_directory(
                                path+'test/img',
                                class_mode=None,
                                target_size=(imSize, imSize),
                                batch_size=batch_size,
                                seed=seed)
    test_mask_datagen = ImageDataGenerator().flow_from_directory(
                                path+'test/gt',
                                class_mode=None,
                                target_size=(imSize, imSize),
                                batch_size=batch_size,
                                color_mode='grayscale',
                                seed=seed)

    train_samples = train_image_datagen.samples
    test_samples = test_image_datagen.samples
    train_generator = imerge(train_image_datagen, train_mask_datagen)
    test_generator = imerge(test_image_datagen, test_mask_datagen)

    return train_generator, test_generator, train_samples, test_samples


  