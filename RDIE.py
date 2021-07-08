# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from keras import Model


def RIE1C(shape=(None,None,1),size=(14,14),levels=16,strides=(1,1),gamma=2.5):
    """ regional information entropy of image with one channel

    :param shape: The image shape(height,width,channels=1).
    :param size: The size of average pooling which decides the receptive field of RIE.
    :param levels: The grayscale levels after quantization.
    :param strides: strides of average pooling, larger strides can reduce calculating amount, but
     will generate rougher results.
    :param gamma: The coefficient of the Gamma transform before quantization.
    :return:a keras model which can calculate the regoinal information entropy map of a grayscale image
    """
    input = tf.keras.Input(shape=shape,dtype=tf.float32)
    #灰度分级
    x = 255 * ((input / 255) ** (1 / gamma))
    x = x//(256//levels)
    multiples = tf.constant([1, 1, 1, levels], dtype=tf.int32)
    x = tf.tile(x,multiples)
    #计算各灰度级的有效值
    bias = tf.range(0,-levels,-1,tf.float32)
    x = tf.nn.bias_add(x,bias)
    x = tf.where(x==0,1.,0.)
    # 统计区域内各灰度级的频率
    x = tf.keras.layers.AveragePooling2D(size,strides=strides)(x)
    #计算区域内的信息熵
    x = tf.reduce_sum(tf.where(x>0,-x*tf.math.log(x),0),-1,True)/(np.log(levels))
    return Model(input,x,name="area_entropy")
def RIE3C(shape=(None,None,3),size=(14,14),levels=16,strides=(1,1),gamma=2.5):
    """regional information entropy of image with three channels

    :param shape: The image shape(height,width,channels=3).
    :param size: The size of average pooling which decides the receptive field of RIE.
    :param levels: The grayscale levels after quantization.
    :param strides: strides of average pooling, larger strides can reduce calculating amount, but
     will generate rougher results.
    :param gamma: The coefficient of the Gamma transform before quantization.
    :return:a keras model which can calculate the regional information entropy map of an RGB image
    """
    input = tf.keras.Input(shape=shape,dtype=tf.float32)
    #灰度分级
    x = 255 * ((input / 255) ** (1 / gamma))
    x = x//(256//levels)
    x = tf.transpose(x,[3,1,2,0])
    multiples = tf.constant([1, 1, 1, levels], dtype=tf.int32)
    x = tf.tile(x,multiples)
    #计算各灰度级的有效值
    bias = tf.range(0,-levels,-1,tf.float32)
    x = tf.nn.bias_add(x,bias)
    x = tf.where(x==0,1.,0.)
    # 统计区域内各灰度级的频率
    x = tf.keras.layers.AveragePooling2D(size,strides=strides)(x)
    #计算区域内的信息熵
    x = tf.reduce_sum(tf.where(x>0,-x*tf.math.log(x),0),-1,True)/(np.log(levels))
    return Model(input,x,name="RegInfoEntropy")

class RDIE(object):
    """Calculate RDIE between test image and reference image (RGB images)
    Arguments:
        size: the receptive field of RIE.
        levels: The grayscale levels after quantization.
        gamma: The coefficient of the Gamma transform before quantization.
    Notes:
        For SR problems, we advise to set size=(14,14),levels=16,gamma=2.5
        For traditional distortion, we advise to set size=(7,7),levels=4,gamma=2.5
    Examples:
        metric = RDIE()
        testImg = cv2.imread('SR.png')
        refImg = cv2.imread('HR.png')
        res = metric.call(testImg,refImg)
    """
    def __init__(self,size=(14,14),levels=16,gamma=2.5):
        self.model = RIE3C(size=size,levels=levels,gamma=gamma)
    def call(self,img1,img2):
        img1 = img1[None,:]
        img2 = img2[None,:]
        res1 = self.model(img1)
        res2 = self.model(img2)
        return tf.reduce_mean((255*(res1-res2))**2)