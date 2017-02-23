from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D
import tensorflow as tf

class UNet():
    def __init__(self):
        print ('build UNet ...')

    def get_crop_shape(self, target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

    def create_model(self, img_shape=None, backend='th', tf_input=None):

        dim_ordering = backend
        if backend == 'tf':
            inputs = tf_input
            concat_axis = 3
        else:
            inputs = Input(shape = img_shape)
            concat_axis = 1
            
        conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering, name='conv1_1')(inputs)
        conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), dim_ordering=dim_ordering)(conv1)
        conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(pool1)
        conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2), dim_ordering=dim_ordering)(conv2)

        conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(pool2)
        conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2), dim_ordering=dim_ordering)(conv3)

        conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(pool3)
        conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2), dim_ordering=dim_ordering)(conv4)

        conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(pool4)
        conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv5)

        up_conv5 = UpSampling2D(size=(2, 2), dim_ordering=dim_ordering)(conv5)
        ch, cw = self.get_crop_shape(conv4, up_conv5)
        crop_conv4 = Cropping2D(cropping=(ch,cw), dim_ordering=dim_ordering)(conv4)
        up6 = merge([up_conv5, crop_conv4], mode='concat', concat_axis=concat_axis)
        conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(up6)
        conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv6)

        up_conv6 = UpSampling2D(size=(2, 2), dim_ordering=dim_ordering)(conv6)
        ch, cw = self.get_crop_shape(conv3, up_conv6)
        crop_conv3 = Cropping2D(cropping=(ch,cw), dim_ordering=dim_ordering)(conv3)
        up7 = merge([up_conv6, crop_conv3], mode='concat', concat_axis=concat_axis)
        conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(up7)
        conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv7)

        up_conv7 = UpSampling2D(size=(2, 2), dim_ordering=dim_ordering)(conv7)
        ch, cw = self.get_crop_shape(conv2, up_conv7)
        crop_conv2 = Cropping2D(cropping=(ch,cw), dim_ordering=dim_ordering)(conv2)
        up8 = merge([up_conv7, crop_conv2], mode='concat', concat_axis=concat_axis)
        conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(up8)
        conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv8)

        up_conv8 = UpSampling2D(size=(2, 2), dim_ordering=dim_ordering)(conv8)
        ch, cw = self.get_crop_shape(conv1, up_conv8)
        crop_conv1 = Cropping2D(cropping=(ch,cw), dim_ordering=dim_ordering)(conv1)
        up9 = merge([up_conv8, crop_conv1], mode='concat', concat_axis=concat_axis)
        conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(up9)
        conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv9)

        ch, cw = self.get_crop_shape(inputs, conv9)
        conv9 = ZeroPadding2D(padding=(ch[0], ch[1], cw[0], cw[1]), dim_ordering=dim_ordering)(conv9)
        conv10 = Convolution2D(1, 1, 1, activation='sigmoid', dim_ordering=dim_ordering)(conv9)

        if backend == 'tf':
            return conv10
        else:
            model = Model(input=inputs, output=conv10)
            return model


