#-*-coding:utf-8-*-

'''
Mobis, Simulator, Udacity가 섞여있는 training-prediction-file에서 Mobis만 뽑아 RMSE를 구하는 코드
ex) python trainRMSE.py train-predictions-epoch6
'''

import os
import sys
import math

fp = open(sys.argv[1], 'r')
fg = open('/media/user/c45eb821-d419-451d-b171-3152a8436ba2/data/drivePX/train/RMSE/TRAIN','r')

angle_gt = [[0.0]*100000 for i in range(2)]
angle_pr = [[0.0]*100000 for i in range(2)]
RMSE = 0.0
cnt = 0

# get GT
while True:
    line = fg.readline()
    if not line: break
    line = line.split(',')
    angle_gt[0 if line[0][:6] == '173750' else 1][int(line[0][-10:-4])] = float(line[1])

# get Mobis train
while True:
    line = fp.readline()
    if not line: break
    if line.find('crop_rgba') == -1: continue
    line = line.split(',')
    cnt = cnt + 1

    A = 0 if os.path.basename(line[0])[:6] == '173750' else 1
    B = int(line[0][-10:-4])
    angle_pr[A][B] = float(line[1])
    RMSE = RMSE + pow((angle_gt[A][B] - angle_pr[A][B]), 2)


RMSE = math.sqrt(RMSE/cnt)
print RMSE, cnt
