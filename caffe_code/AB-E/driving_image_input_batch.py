#---------------------------------------------------------
#Python Layer for the cityscapes dataset

# Adapted from: Fully Convolutional Networks for Semantic Segmentation by Jonathan Long*, Evan Shelhamer*, and Trevor Darrell. CVPR 2015 and PAMI 2016. http://fcn.berkeleyvision.org


#---------------------------------------------------------

import caffe
import numpy as np
from PIL import Image
import os
import pdb
import random
import re
import math

class input_layer(caffe.Layer):
    """
    Load (input image, label image) pairs from the SBDD extended labeling
    of PASCAL VOC for semantic segmentation
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - cityscapes_dir: path to SBDD `dataset` dir
        - split: train / seg11valid
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for SBDD semantic segmentation.

        N.B.segv11alid is the set of segval11 that does not intersect with SBDD.
        Find it here: https://gist.github.com/shelhamer/edb330760338892d511e.

        example

        params = dict(cityscapes_dir="/path/to/SBDD/dataset",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="valid")
        """

        # config
        params = eval(self.param_str)
        self.channel6_dir = params['input_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.resize = np.array(params['resize'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.batch_size = params.get('batch_size', None)
        self.idx = np.array(range(self.batch_size))
        self.batch = np.array([self.batch_size]*self.batch_size)
        #self.idx = [324]
        self.path = '/media/ys/1a32a0d7-4d1f-494a-8527-68bb8427297f/End_to_End/'
        label_path4 = '/media/ys/1a32a0d7-4d1f-494a-8527-68bb8427297f/End_to_End/Mobis_crop/TRAIN1.txt'

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        train_data = []
        filename4 = open(label_path4, 'r')
        patten4 = '(\w*/\w*/\w*.\w*),([-\d]*.\d*)'
        r4 = re.compile(patten4)

        summation = 0
        summation_sqr = 0

        while True:
            line = filename4.readline()
            line_split = r4.findall(line)
            if not line: break
            train_data.append(line_split[0])
 #           pdb.set_trace()
            summation += float(line_split[0][1])
            summation_sqr += (float(line_split[0][1])) * (float(line_split[0][1]))
        filename4.close()
        self.train_data = train_data
        #pdb.set_trace()
        if self.random:
            random.seed(self.seed)
            self.idx = random.sample(range(len(self.train_data)),self.batch_size)
        length = len(self.train_data)
        mean = summation/length
        sqr_mean = summation_sqr/length
        var = sqr_mean - mean * mean
        std = math.sqrt(var)
        self.mean_label = mean
        self.sqr_mean = sqr_mean
        self.std = std        
        #pdb.set_trace()
        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

    def reshape(self, bottom, top):
        # load image + label image pair
        #pdb.set_trace()
        self.data, self.label = self.load_batch(self.idx, self.batch_size)
        #pdb.set_trace()
        # reshape tops to f~/it (leading 1 is for batch dimension)      
        top[0].reshape(*self.data.shape)
        top[1].reshape(*self.label.shape)
    def forward(self, bottom, top):
#        pdb.set_trace()
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        # pick next input
        #pdb.set_trace()
        if self.random:
            self.idx = random.sample(range(len(self.train_data)),self.batch_size)
        else:
        	#self.idx = self.idx
        	self.idx = self.idx + self.batch
        	#pdb.set_trace()
        	if self.idx[-1] > len(self.train_data) - self.batch_size:
        		self.idx = np.array(range(self.batch_size))
        #pdb.set_trace()


    def backward(self, top, propagate_down, bottom):
        #pdb.set_trace()
        pass

    def load_batch(self, idx, batch_size):
        batch_im = np.zeros((self.batch_size,3,100,250), dtype = np.float32)
        for i in range(self.batch_size):
#            pdb.set_trace()
            batch_im[i] = self.load_image_label(self.path+'Mobis_crop/'+self.train_data[self.idx[i]][0])
#        pdb.set_trace()
        angle = np.zeros((self.batch_size,1), dtype = np.float32)
        for i in range(self.batch_size):
            angle[i] = float(self.train_data[self.idx[i]][1])
#            angle[i] = float(self.train_data[self.idx[i]][1])/(180/np.pi)
 #           angle[i] = (float(self.train_data[self.idx[i]][1])-self.mean_label)/self.std
 #       pdb.set_trace()
        return batch_im, angle
    def load_image_label(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        #pdb.set_trace()
        im = Image.open(idx)
#        in_ori = np.array(im, dtype=np.float32)
#        pdb.set_trace()
        im = im.resize([self.resize[1],self.resize[0]],Image.ANTIALIAS)
        in_ = np.array(im, dtype=np.float32)
#        in_ = -1.0 + 2.0 * in_/255.0
#        pdb.set_trace()
        in_ = in_[:,:,::-1]
        in_ = in_.transpose((2,0,1))
        return in_