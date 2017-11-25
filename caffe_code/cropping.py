import cv2
import pdb
import numpy as np
import re
import os

path = '/media/ys/HU/mobis_20171123/seosan/'
folder_name = '1c'
data_path = path+folder_name+'/'
for img_idx in range(len(os.listdir(data_path))):
#    img_idx = 1093
    img = cv2.imread(data_path+folder_name+'_center_screenshot_rgba_'+'%06d.png'%(img_idx))
#    pdb.set_trace()
    crop_img = img[250:500,150:810]
#    pdb.set_trace()
#    cv2.rectangle(img, (150,250),(810,500),(0,0,255),2)
    img_name = (path+'../test_road/'+folder_name+'/'+'crop_image'+folder_name+'_center_crop_%06d.png'%(img_idx))
    print img_idx
    cv2.imwrite(img_name, crop_img)
    pdb.set_trace()