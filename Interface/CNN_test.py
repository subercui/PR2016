# -*- coding: utf-8 -*-

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint,Callback
import theano

import numpy as np
import cPickle,os,argparse,sys

import PR2016,utils
from matplotlib import pyplot as plt
from scipy.misc import imread, imresize, imsave

def process_args(args, description):
    """
    Handle the command line.
    args     - list of command line arguments (not including executable name)
    defaults - a name space with variables corresponding to each of
               the required default command line values.
    description - a string to display at the top of the help message.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-I', '--image', dest="img_path", type=str,
                        help='image relative path, eg ./test.jpg (default: None')
    parameters = parser.parse_args(args)

    return parameters
def test():
	parameters = process_args(sys.argv[1:], __doc__)

	datadir = os.path.split(os.path.realpath(__file__))[0]
	#print 'data dir',datadir
	batch_size = 32
	nb_classes = 7
	nb_epoch = 30
	data_augmentation = True

	# input image dimensions
	img_size = 250
	img_rows, img_cols = img_size, img_size
	# the CIFAR10 images are RGB
	img_channels = 3

	#build model
	model = model_from_json(open(datadir+'/model.json').read())
	model.load_weights(datadir+'/weights.29-0.96.hdf5')

	img = imread(parameters.img_path)#1000*750,默认长边在前才是正方向
	if img.shape[0]<img.shape[1]:
		img=img.transpose(1,0,2)
	img=imresize(img, (img_size,img_size)).transpose(2,0,1)
	              
	img=img[None,:]

	result=model.predict(img)
	result=np.argmax(result.flatten())
	print 'the image is belong to class:',result
	return result

result=test()