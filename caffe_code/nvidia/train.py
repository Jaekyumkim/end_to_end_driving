import caffe
import matplotlib.pyplot as plt
import time as timelib
import pdb

caffe.set_mode_gpu()
caffe.set_device(1)

solver = caffe.get_solver('/media/ys/1a32a0d7-4d1f-494a-8527-68bb8427297f/End_to_End/caffe/solver.prototxt')
#solver.net.copy_from('/media/ys/1a32a0d7-4d1f-494a-8527-68bb8427297f/End_to_End/caffe/weight/nvidia/nvidia_00001_iter_5000.caffemodel')
#solver.net.copy_from('/media/ys/1a32a0d7-4d1f-494a-8527-68bb8427297f/Data/train/checkpoint-sdc-ch2.data-00000-of-00001.caffemodel')
max_iter = 800000
fig, axes = plt.subplots()
fig.show()
loss = 0
loss_list = []
iter0 = solver.iter
epoch = 0

while solver.iter < max_iter:

#  net_full_conv.save('./copy_vgg.caffemodel')
  solver.step(1)
  #if solver.iter == 3000:
  #	pdb.set_trace()
#  if solver.iter % 100 == 0:
#  	pdb.set_trace()
  #pdb.set_trace()
  #if solver.iter % 500 ==0:
  label = solver.net.blobs['label'].data 
  out = solver.net.blobs['fc10'].data
  #pdb.set_trace()
  loss = solver.net.blobs['loss'].data.flatten()
  if loss > 30:
  	loss = 30
  loss_list.append(loss)  
  if solver.iter % 100 == 0:
    axes.clear()
    axes.plot(range(iter0, iter0+len(loss_list)), loss_list)
#    axes.grid(True)
    fig.canvas.draw()
    plt.pause(0.01)

fig.savefig('fig_iter_%d.png' % solver.iter)