import caffe
import matplotlib.pyplot as plt
from PIL import Image
import time as timelib
import pdb
import numpy as np
import os
import sys
import cv2
import time
import glob
import re
import scipy.io as si
import copy
import math

input_WIDTH = 610
input_HEIGHT = 250
test_WIDTH = 250
test_HEIGHT = 100
caffe.set_mode_gpu()
caffe.set_device(1)
net = caffe.Net('./deploy_abe.prototxt', './weight/ABE_00001/ABE_00001_iter_165000.caffemodel', caffe.TEST)
phase = 'test'
writer = cv2.VideoWriter('./demo/mobis_abe_'+phase+''+'.avi', cv2.cv.CV_FOURCC(*'DIVX'), 20, (input_WIDTH, input_HEIGHT))
std = 5.7385705
mean = 0.46000889
RMSE = 0

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.array([104,117,123]))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)
net.blobs['data'].reshape(1,3,test_HEIGHT,test_WIDTH)
basedir = '/media/ys/1a32a0d7-4d1f-494a-8527-68bb8427297f/End_to_End/Mobis_crop/'
if phase == 'train':
    label_path = '/media/ys/1a32a0d7-4d1f-494a-8527-68bb8427297f/End_to_End/Mobis_crop/TRAIN.txt'
    filename = open(label_path, 'r')
    patten = '(\w*/\w*/\w*.\w*),([-\d]*.\d*)'
    r = re.compile(patten)
    label = []
    while True:
        line = filename.readline()
        line_split = r.findall(line)
        if not line: break
        label.append(line_split[0])
    for i in range(len(label)):
    	#pdb.set_trace()
        frame = Image.open(basedir+label[i][0])
        frame = frame.resize([test_WIDTH,test_HEIGHT],Image.ANTIALIAS)
        frame = np.array(frame, dtype=np.float32)
#        frame = -1.0 + 2.0 * frame/255.0
        frame = frame[:,:,::-1]
        frame = frame.transpose((2,0,1))
        frame_video = cv2.imread(basedir+label[i][0])
        net.blobs['data'].data[...] = frame
        out = net.forward()
#        prediction_angle = (out['fc10']*std + mean)/(180/np.pi)
        prediction_angle = (out['fc10'])/(180/np.pi)
        frame_gt = float(label[i][1])/(180/np.pi)
        diff = (float(out['fc10'][0][0])-float(label[i][1]))*(float(out['fc10'][0][0])-float(label[i][1]))
        RMSE += diff
        cv2.rectangle(frame_video, (10, 210), (200, 245), (255,255,255), -1)
        cv2.putText(frame_video, "predicted:    " + str(round(prediction_angle*(180/np.pi),4)), (10,220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
        cv2.putText(frame_video, "ground_truth:" + str(round(frame_gt*(180/np.pi),4)), (10,240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
        rad = 50
        cv2.circle(frame_video, (305, 200), rad, (255, 255, 255), 1)
        x1 = rad * np.cos(np.pi/2 + frame_gt)
        y1 = rad * np.sin(np.pi/2 + frame_gt)
        cv2.circle(frame_video, (305 + int(x1), 200 - int(y1)), 7, (0,0,255), -1)
        x2 = rad * np.cos(np.pi/2 + prediction_angle)
        y2 = rad * np.sin(np.pi/2 + prediction_angle)
        cv2.circle(frame_video, (305 + int(x2), 200 - int(y2)), 4, (255,0,0),  -1)
        writer.write(frame_video)
        print i
        cv2.waitKey(1)
elif phase == 'test':
    label_path = '/media/ys/1a32a0d7-4d1f-494a-8527-68bb8427297f/End_to_End/Mobis_crop/TEST.txt'
    filename = open(label_path, 'r')
    patten = '(\w*/\w*/\w*.\w*),([-\d]*.\d*)'
    r = re.compile(patten)
    label = []
    while True:
        line = filename.readline()
        line_split = r.findall(line)
        if not line: break
        label.append(line_split[0])
    for i in range(len(label)):
        #pdb.set_trace()
        frame = Image.open(basedir+label[i][0])
        frame = frame.resize([test_WIDTH,test_HEIGHT],Image.ANTIALIAS)
        frame = np.array(frame, dtype=np.float32)
#        frame = -1.0 + 2.0 * frame/255.0
        frame = frame[:,:,::-1]
        frame = frame.transpose((2,0,1))
        frame_video = cv2.imread(basedir+label[i][0])
        net.blobs['data'].data[...] = frame
        out = net.forward()
#        prediction_angle = (out['fc10']*std + mean)/(180/np.pi)
        prediction_angle = (out['fc10'])/(180/np.pi)
        frame_gt = float(label[i][1])/(180/np.pi)
        diff = (float(out['fc10'][0][0])-float(label[i][1]))*(float(out['fc10'][0][0])-float(label[i][1]))
        RMSE += diff
        #pdb.set_trace()
        cv2.rectangle(frame_video, (10, 210), (200, 245), (255,255,255), -1)
        cv2.putText(frame_video, "predicted:    " + str(round(prediction_angle*(180/np.pi),4)), (10,220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
        cv2.putText(frame_video, "ground_truth:" + str(round(frame_gt*(180/np.pi),4)), (10,240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
        rad = 50
        cv2.circle(frame_video, (305, 200), rad, (255, 255, 255), 1)
        x1 = rad * np.cos(np.pi/2 + frame_gt)
        y1 = rad * np.sin(np.pi/2 + frame_gt)
        cv2.circle(frame_video, (305 + int(x1), 200 - int(y1)), 7, (0,0,255), -1)
        x2 = rad * np.cos(np.pi/2 + prediction_angle)
        y2 = rad * np.sin(np.pi/2 + prediction_angle)
        cv2.circle(frame_video, (305 + int(x2), 200 - int(y2)), 4, (255,0,0),  -1)
        writer.write(frame_video)
        print i
        cv2.waitKey(1)
else:
    sys.exit("Error message")

RMSE = math.sqrt(RMSE)/i
print phase + '_RMSE = ' + str(RMSE)
writer.release()
cv2.destroyAllWindows()
