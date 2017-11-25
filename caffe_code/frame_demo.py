import caffe
import matplotlib.pyplot as plt
import time as timelib
import pdb
import numpy as np
import os
import sys
import cv2
import time
import glob

base_dir = '/media/ys/HU/mobis_20171123/test_road/'
folder_name = '3c_2'
f_new = open(base_dir+folder_name+'/label/label.txt','r')
data_path = base_dir+folder_name+'/original_image/center/'

input_WIDTH = 960
input_HEIGHT = 604
writer = cv2.VideoWriter(base_dir+folder_name+'/'+folder_name+'_demo.avi', cv2.cv.CV_FOURCC(*'DIVX'), 20, (input_WIDTH, input_HEIGHT))
num = 0
rad = 50
while True:
    line = f_new.readline()
    if not line: break
    steering_angle = line.split(',')[1]
#    pdb.set_trace()
    frame_video = cv2.imread(data_path+folder_name+'_center_screenshot_rgba_'+'%06d.png'%(num))
    frame_gt = float(steering_angle)/(180/np.pi)
    cv2.circle(frame_video, (60, 320), rad, (255, 255, 255), 1)
    x1 = rad * np.cos(np.pi/2 + frame_gt)
    y1 = rad * np.sin(np.pi/2 + frame_gt)
    cv2.circle(frame_video, (60 + int(x1), 320 - int(y1)), 7, (0,0,255), -1)
    cv2.putText(frame_video, "frame:" + str(num), (10,240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
    cv2.rectangle(frame_video, (150, 250), (810, 500), (0,0,255), 2)
    writer.write(frame_video)
    print num
    num += 1
    cv2.waitKey(1)

