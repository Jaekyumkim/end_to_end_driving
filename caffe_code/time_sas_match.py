import pdb

base_dir = '/media/ys/HU/mobis_20171123/test_road/'
folder_name = '3c_2'
f_can = open(base_dir+folder_name+'/can_data/'+folder_name+'.txt','r')
f_video = open(base_dir+folder_name+'/etc/video_time_0.txt','r')
f_new = open(base_dir+folder_name+'/label/label.txt','w')

line = f_video.readline()
line2 = f_can.readline()
line2temp = line2

match = []
while True:
#    pdb.set_trace()
    if int(line.split(' ')[1][:-1]) < int(line2.split(',')[0]):
        if abs(int(line.split(' ')[1][:-1]) - int(line2.split(',')[0])) < abs(int(line.split(' ')[1][:-1]) - int(line2temp.split(',')[0])):
            match.append(float(line2.split(',')[1][:-1]))
        else :
            match.append(float(line2temp.split(',')[1][:-1]))
        line = f_video.readline()
    line2temp = line2
    line2 = f_can.readline()
    #pdb.set_trace()
    if not line or not line2: break
for i in range(len(match)):
    data = '/media/ys/HU/mobis_20171123/test_road/'+folder_name+'/crop_image/center/'+folder_name+'_center_crop_'+'%06d.png,%f\n'%(i,match[i])
    f_new.write(data)
