###########################################################################################
# Code for Using CNN fully connected layed as the feature
#------------------------------------------------------------------------------------------

import numpy as np
# np.set_printoptions(threshold='nan')
import cv2
import matplotlib.pyplot as plt
# display plots in this notebook
# %matplotlib inline
import sys
caffe_root = '/home/student/Documents/PSPNet/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
caffe.set_mode_cpu()

filenames = []
for i in range(0,len(res)):
    name = "/home/student/Documents/Living_Indicator/img/file"+str(i)+".png"
    filenames.append(name)

model_def = '/home/student/Documents/caffe/models/bvlc_alexnet/deploy.prototxt'
model_weights = '/home/student/Documents/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel'

net = caffe.Net(model_def,  # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)

mu = np.load('/home/student/Documents/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

count = 0
feat = np.empty(0)

for i in range(0, len(filenames)):
    image = caffe.io.load_image(filenames[i])
    transformed_image = transformer.preprocess('data', image)

    net.blobs['data'].data[...] = transformed_image
    output = net.forward()

    temp = net.blobs['fc7'].data[:, :]

    if (count == 0):
        feat = temp
    else:
        feat = np.vstack((feat, temp))

    count = count + 1
    print(count)

    del temp

print(feat.shape)