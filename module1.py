import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal, TruncatedNormal
import tensorflow as tf

class BarEncoder(Model):
    def conv(self, x, f, k, s, a, p, bn):
        x = Conv2D(filters=f
                    , kernel_size=k
                    , padding=p
                    , strides=s
                    ,kernel_initializer=TruncatedNormal(stddev=0.02))(x)
        if bn:
            x = BatchNormalization(momentum=0.9, epsilon=1e-5, scale=True)(x)
        if a == 'relu':
            x = Activation(a)(x)
        elif a == 'lrelu':
            x = LeakyReLU(alpha=0.2)(x)
        return x

    def __init__(self, *args, **kwargs):
        super(BarEncoder, self).__init__(*args, **kwargs)
        self.conv = self.conv
    
    def call(self, cond_tensor):
        h0 = self.conv(cond_tensor, f=16, k=[1, 12], s=[1, 12], a='lrelu', p='valid', bn=True)
        h1 = self.conv(h0, f=16, k=[1, 7], s=[1, 7], a='lrelu', p='valid', bn=True)
        h2 = self.conv(h1, f=16, k=[3, 1], s=[3, 1], a='lrelu', p='valid', bn=True)
        h3 = self.conv(h2, f=16, k=[2, 1], s=[2, 1], a='lrelu', p='valid', bn=True)
        h4 = self.conv(h3, f=16, k=[2, 1], s=[2, 1], a='lrelu', p='valid', bn=True)
        h5 = self.conv(h4, f=16, k=[2, 1], s=[2, 1], a='lrelu', p='valid', bn=True)
        h6 = self.conv(h5, f=16, k=[2, 1], s=[2, 1], a='lrelu', p='valid', bn=True)

        return [h0, h1, h2, h3, h4, h5, h6]

class BarGenerator(Model):
    def conv_t(self, x, f, k, s, a, p, bn):
        x = Conv2DTranspose(filters=f
                            , kernel_size=k
                            , strides=s
                            , padding=p
                            , kernel_initializer=TruncatedNormal(stddev=0.02))(x)
        if bn:
            x = BatchNormalization(momentum=0.9, epsilon=1e-5, scale=True)(x)
        if a=='relu':
            x = Activation(a)(x)
        elif a=='relu':
            x = LeakyReLU(alpha=0.2)(x)
        return x
    def concat_prev(self, tensor_in, condition):
        if condition is None:
            return tensor_in
        else:
            if tensor_in.get_shape()[1:3] == condition.get_shape()[1:3]:
                return Concatenate(axis=3)([tensor_in, condition])
            else:
                raise ValueError('MisMatch', tensor_in.shape, '!=', condition.shape, '!!!!')
    
    def __init__(self):
        super(BarGenerator, self).__init__()
        self.conv_t = self.conv_t
        self.concat_prev = self.concat_prev
    
    def call(self, in_tensor, nowbar=None):
        x0 = tf.reshape(in_tensor, tf.stack([-1, 1, 1, in_tensor.get_shape()[1]]))
        x0 = self.conv_t(x0, f=1024, k=[1, 1], s=[1, 1], a='relu', p='valid', bn=True)
        
        x1 = Reshape([2, 1, 512])(x0) # 2,1,512 -> 1,1,512
        x1 = self.concat_prev(x1, nowbar[6])
        x1 = self.conv_t(x1, f=512, k=[2, 1], s=[2, 1], a='relu', p='valid', bn=True)

        x2 = self.concat_prev(x1, nowbar[5])
        x2 = self.conv_t(x2, f=256, k=[2, 1], s=[2, 1], a='relu', p='valid', bn=True)

        x3 = self.concat_prev(x2, nowbar[4])
        x3 = self.conv_t(x3, f=256, k=[2, 1], s=[2, 1], a='relu', p='valid', bn=True)

        x4 = self.concat_prev(x3, nowbar[3])
        x4 = self.conv_t(x4, f=128, k=[2, 1], s=[2, 1], a='relu', p='valid', bn=True)

        x5 = self.concat_prev(x4, nowbar[2])
        x5 = self.conv_t(x5, f=128, k=[3, 1], s=[3, 1], a='relu', p='valid', bn=True)

        x6 = self.concat_prev(x5, nowbar[1])
        x6 = self.conv_t(x6, f=64, k=[1, 7], s=[1, 1], a='relu', p='valid', bn=True)

        x7 = self.concat_prev(x6, nowbar[0])
        x7 = self.conv_t(x7, f=1, k=[1, 12], s=[1, 12], a='tanh', p='valid', bn=False)

        return x7