##############################################################################################################
#    code for saving segmented images (car, house, road, tree, terrain etc)
##############################################################################################################

import numpy as np
import cv2
import sys

caffe_root = '/home/student/Documents/PSPNet'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python/')

import caffe


import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# caffe.set_mode_cpu()
model_def = '/home/cv/Sumukh/PSPNet/evaluation/prototxt/pspnet101_cityscapes_713.prototxt'
model_weights = '/home/cv/Sumukh/PSPNet/caffemodel/pspnet101_cityscapes.caffemodel'

net = caffe.Net(model_def,  # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)  # use test mode (e.g., don't perform dropout)

mu = np.load('/home/cv/Sumukh/PSPNet/python/caffe/imagenet/ilsvrc_2012_mean.npy')
# mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(1,  # batch size
                          3,  # 3-channel (BGR) images
                          713, 713)  # image size is 713*713
for i in range(5653, 5654): #5464
    filename = 'file' + str(i) + '.png'
    image = caffe.io.load_image('/home/cv/Sumukh/Living_Indicator/img/' + filename)
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    output = net.forward()

    img1 = net.blobs['conv6_interp'].data[0, 0]  # road
    img2 = net.blobs['conv6_interp'].data[0, 2]  # house
    img3 = net.blobs['conv6_interp'].data[0, 8]  # trees
    img4 = net.blobs['conv6_interp'].data[0, 9]  # terrain
    img5 = net.blobs['conv6_interp'].data[0, 13]  # Car

    ret, th1 = cv2.threshold(img1, 6, 255, cv2.THRESH_BINARY)
    ret, th2 = cv2.threshold(img2, 9, 255, cv2.THRESH_BINARY)
    ret, th3 = cv2.threshold(img3, 6, 255, cv2.THRESH_BINARY)
    ret, th4 = cv2.threshold(img4, 5, 255, cv2.THRESH_BINARY)
    ret, th5 = cv2.threshold(img5, 5, 255, cv2.THRESH_BINARY)

    plt.imshow(th1)
    plt.savefig('/home/cv/Sumukh/Results/Road/test_road' + str(i) + '.png')
    plt.imshow(th2)
    plt.savefig('/home/cv/Sumukh/Results/House/test_house' + str(i) + '.png')
    plt.imshow(th3)
    plt.savefig('/home/cv/Sumukh/Results/Trees/test_tree' + str(i) + '.png')
    plt.imshow(th4)
    plt.savefig('/home/cv/Sumukh/Results/Terrain/test_terrain' + str(i) + '.png')
    plt.imshow(th5)
    plt.savefig('/home/cv/Sumukh/Results/Car/test_car' + str(i) + '.png')