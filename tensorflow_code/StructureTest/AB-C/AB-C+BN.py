'''
A

Tensor("Vision/BatchNorm_1/batchnorm/add_1:0", shape=(16, 100, 250, 64), dtype=float32)
Tensor("Vision/MaxPool2D/MaxPool:0", shape=(16, 50, 125, 64), dtype=float32)
Tensor("Vision/MaxPool2D_1/MaxPool:0", shape=(16, 25, 62, 128), dtype=float32)
Tensor("Vision/MaxPool2D_2/MaxPool:0", shape=(16, 12, 31, 256), dtype=float32)
Tensor("Vision/MaxPool2D_3/MaxPool:0", shape=(16, 6, 15, 512), dtype=float32)
B

Tensor("Vision/Conv_9/Relu:0", shape=(16, 3, 8, 512), dtype=float32)
Tensor("Vision/Conv2d_transpose/Relu:0", shape=(16, 6, 16, 512), dtype=float32)
Tensor("Vision/Conv_11/Relu:0", shape=(16, 2, 4, 256), dtype=float32)
Tensor("Vision/Conv2d_transpose_1/Relu:0", shape=(16, 12, 32, 512), dtype=float32)
Tensor("Vision/Conv_13/Relu:0", shape=(16, 1, 2, 256), dtype=float32)
Tensor("Vision/Conv2d_transpose_2/Relu:0", shape=(16, 24, 64, 512), dtype=float32)
'''
# coding: utf-8

# This jupyter notebook describes the solution approach of the team komanda to udacity self-driving car challenge 2: predicting steering angle from images.
# 
# Author: Ilya Edrenkin, `ilya.edrenkin@gmail.com`

# In[ ]:

import tensorflow as tf
from tensorflow.python.util import nest
import numpy as np
import os
slim = tf.contrib.slim
import pdb
import pickle
import time


# # Outline
# 
# The presented model performs a mapping from sequences of images to sequences of steering angle measurements. The mapping is causal, i.e. there is no "looking into future" -- only past frames are used to predict the future steering decisions.
# 
# The model is based on three key components:
# 1) The input image sequences are processed with a 3D convolution stack, where the discrete time axis is interpreted as the first "depth" dimension. That allows the model to learn motion detectors and understand the dynamics of driving.
# 2) The model predicts not only the steering angle, but also the vehicle speed and the torque applied to the steering wheel. 
# 3) The model is stateful: the two upper layers are a LSTM and a simple RNN, respectively. The predicted angle, torque and speed serve as the input to the next timestep. 
#  
# The model is optimized jointly for the autoregressive and ground truth modes: in the former, model's own outputs are fed into next timestep, in the latter, real targets are used as the context. Naturally, only autoregressive mode is used at the test time.
# 
# I used a single GTX 1080 to train the model. In the training phase there was a constraint to fit into the memory of the card (8 GB). For the evaluation phase the model was performing nearly twice as fast as real-time in this setup.
# 
# Data extraction from rosbags is performed using Ross Wightman's scripts, because these were also used for the test data in this challenge; for real-life scenarios (and not for the challenge) it would make sense to read data directly into the model from the rosbags. Another concern about real-life is that the steering angle sequence that is to be predicted should be probably delayed by the actuator's latency.
# 
# No data augmentation (except for aggressive regularization via dropout) is used.

# In[ ]:

# define some constants

# RNNs are typically trained using (truncated) backprop through time. SEQ_LEN here is the length of BPTT. 
# Batch size specifies the number of sequence fragments used in a sigle optimization step.
# (Actually we can use variable SEQ_LEN and BATCH_SIZE, they are set to constants only for simplicity).
# LEFT_CONTEXT is the number of extra frames from the past that we append to the left of our input sequence.
# We need to do it because 3D convolution with "VALID" padding "eats" frames from the left, decreasing the sequence length.
# One should be careful here to maintain the model's causality.
SEQ_LEN = 2
BATCH_SIZE = 8
LEFT_CONTEXT = 0

# These are the input image parameters.
HEIGHT = 100
WIDTH = 250
CHANNELS = 3 # RGB

# The parameters of the LSTM that keeps the model state.
RNN_SIZE = 32
RNN_PROJ = 32

# Our training data follows the "interpolated.csv" format from Ross Wightman's scripts.
CSV_HEADER = "index,timestamp,width,height,frame_id,filename,angle,torque,speed,lat,long,alt".split(",")
OUTPUTS = CSV_HEADER[-6:-3] # angle,torque,speed
OUTPUT_DIM = 1 # predict all features: steering angle, torque and vehicle speed


# # Input/output format
# 
# Our data is presented as a long sequence of observations (several concatenated rosbags).
# We need to chunk it into a number of batches: for this, we will create BATCH_SIZE cursors. Let their starting points be uniformly spaced in our long sequence. We will advance them by SEQ_LEN at each step, creating a BATCH_SIZE x SEQ_LEN matrix of training examples.
# Boundary effects when one rosbag ends and the next starts are simply ignored.
# 
# (Actually, LEFT_CONTEXT frames are also added to the left of the input sequence; see code below for details).

# In[ ]:

class BatchGenerator(object):
    def __init__(self, sequence, seq_len, batch_size): # sequences : csv file
        self.sequence = sequence
        self.seq_len = seq_len
        self.batch_size = batch_size
        chunk_size = 1 + (len(sequence) - 1) / batch_size
        self.indices = [(i*chunk_size) % len(sequence) for i in range(batch_size)]
            
    def next(self):
        while True:
            output = []
            for i in range(self.batch_size):
                LEFT_CONTEXT = 5
                idx = self.indices[i]
                left_pad = self.sequence[idx - LEFT_CONTEXT:idx]
                if len(left_pad) < LEFT_CONTEXT:
                    left_pad = [self.sequence[0]] * (LEFT_CONTEXT - len(left_pad)) + left_pad
                assert len(left_pad) == LEFT_CONTEXT
                leftover = len(self.sequence) - idx
                if leftover >= self.seq_len:
                    result = self.sequence[idx:idx + self.seq_len]
                else:
                    result = self.sequence[idx:] + self.sequence[:self.seq_len - leftover]
                assert len(result) == self.seq_len
                self.indices[i] = (idx + self.seq_len) % len(self.sequence)
                images, targets = zip(*result)
                images_left_pad, _ = zip(*left_pad)
                output.append((np.stack(images), np.stack(targets)))
            output = zip(*output)
            output[0] = np.stack(output[0]) # batch_size x (LEFT_CONTEXT + seq_len)
            output[1] = np.stack(output[1]) # batch_size x seq_len x OUTPUT_DIM
            return output
        
# "index,timestamp,width,height,frame_id,[filename,angle, torque,speed],lat,long,alt"

def read_csv(filename):
    lines = []
    #pdb.set_trace()
    with open(filename, 'r') as f:
        while True:
            ln = f.readline()
            if not ln: break
            ln = ln.strip().split(",")
            lines.append([str(filename[:-5] + "../../" + ln[0]), ln[1]])

        lines = map(lambda x: (x[0], np.float32(x[1:])), lines)
        return lines

def readt_csv(filename):
    lines = []
    with open(filename, 'r') as f:
        while True:
            ln = f.readline()
            if not ln: break
            ln = ln.strip().split(",")
            lines.append([str(filename[:-13] + "../../" + ln[0]), ln[1]])
            #pdb.set_trace()
        lines = map(lambda x: (x[0], np.float32(x[1:])), lines)
        return lines

def process_csv(filename, val=5):
    sum_f = np.float128([0.0] * OUTPUT_DIM)
    sum_sq_f = np.float128([0.0] * OUTPUT_DIM)
    lines = read_csv(filename)
    # leave val% for validation 
    train_seq = []
    valid_seq = []
    cnt = 0
    for ln in lines:
        
        if cnt < SEQ_LEN * BATCH_SIZE * (100 - val): 
            train_seq.append(ln)
            sum_f += ln[1]
            sum_sq_f += ln[1] * ln[1]
        else:
            valid_seq.append(ln)
        cnt += 1
        cnt %= SEQ_LEN * BATCH_SIZE * 100
    mean = sum_f / len(train_seq)
    var = sum_sq_f / len(train_seq) - mean * mean
    std = np.sqrt(var)
    print len(train_seq), len(valid_seq)
    print mean, std # we will need these statistics to normalize the outputs (and ground truth inputs)
    return (train_seq, valid_seq), (mean, std)


# In[ ]:

(train_seq, valid_seq), (mean, std) = process_csv(filename="../../../../data/drivePX/train/RMSE/TRAIN", val=5) # concatenated interpolated.csv from rosbags 
test_seq = readt_csv("../../../../data/drivePX/train/RMSE/TESTwithTRAIN") # interpolated.csv for testset filled with dummy values
#pdb.set_trace() 


# # Key tricks
# 
# Now we are ready to build the model.
# In the next cell we will define the vision module and the recurrent stateful cell.
# 
# The vision module takes a tensor of shape `[BATCH_SIZE, LEFT_CONTEXT + SEQ_LEN, HEIGHT, WIDTH, CHANNELS]` and outputs a tensor of shape `[BATCH_SIZE, SEQ_LEN, 128]`. The entire LEFT_CONTEXT is eaten by the 3D convolutions. Well-known tricks like residual connections and layer normalization are used to improve the convergence of the vision module. Dropout between each pair of layers serves as a regularizer.
# 
# We also need to define our own recurrent cell because we need to train our model jointly in two conditions: when it uses ground truth history and when it uses its own past predictions as the context for the future predictions.
# 
# In addition, we define two helper functions: a layer normalizer with trainable gain/offset and a gradient-clipping optimizer.

# In[ ]:

batch_norm = lambda x: tf.contrib.layers.batch_norm(inputs=x, center=True, scale=True, activation_fn=None, trainable=True)

def get_optimizer(loss, lrate):
    optimizer = tf.train.AdamOptimizer(learning_rate=lrate)
    gradvars = optimizer.compute_gradients(loss)
    gradients, v = zip(*gradvars)
    print [x.name for x in v]
    gradients, _ = tf.clip_by_global_norm(gradients, 15.0)
    return optimizer.apply_gradients(zip(gradients, v))

def apply_vision_simple(image, keep_prob, batch_size, seq_len, scope=None, reuse=None):
    video = tf.reshape(image, shape=[batch_size * (LEFT_CONTEXT + seq_len), HEIGHT, WIDTH, CHANNELS])
    with tf.variable_scope(scope, 'Vision', [image], reuse=reuse):
        # A
        net = slim.convolution(video, num_outputs=64, kernel_size=[3,3], stride=[1,1], padding="SAME")
        net = batch_norm(net)
        net = slim.convolution(net, num_outputs=64, kernel_size=[3,3], stride=[1,1], padding="SAME")
        net = batch_norm(net)
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=[2,2])
        net = slim.convolution(net, num_outputs=128, kernel_size=[3,3], stride=[1,1], padding="SAME")
        net = batch_norm(net)
        net = slim.convolution(net, num_outputs=128, kernel_size=[3,3], stride=[1,1], padding="SAME")
        net = batch_norm(net)
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=[2,2])
        net = slim.convolution(net, num_outputs=256, kernel_size=[3,3], stride=[1,1], padding="SAME")
        net = batch_norm(net)
        net = slim.convolution(net, num_outputs=256, kernel_size=[3,3], stride=[1,1], padding="SAME")
        net = batch_norm(net)
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=[2,2])
        net = slim.convolution(net, num_outputs=512, kernel_size=[3,3], stride=[1,1], padding="SAME")
        net = batch_norm(net)
        net = slim.convolution(net, num_outputs=512, kernel_size=[3,3], stride=[1,1], padding="SAME")
        net = batch_norm(net)
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=[2,2])
        chk1 = net
        # B'
        net = slim.convolution(net, num_outputs=256, kernel_size=[1,1], stride=[1,1], padding="SAME")
        chk2 = net = slim.convolution(net, num_outputs=512, kernel_size=[3,3], stride=[2,2], padding="SAME")
        net = slim.convolution(net, num_outputs=128, kernel_size=[1,1], stride=[1,1], padding="SAME")
        chk3 = net = slim.convolution(net, num_outputs=256, kernel_size=[3,3], stride=[2,2], padding="SAME")
        net = slim.convolution(net, num_outputs=128, kernel_size=[1,1], stride=[1,1], padding="SAME")
        chk4 = net = slim.convolution(net, num_outputs=256, kernel_size=[3,3], stride=[2,2], padding="SAME") # 1, 2

        # deconv
        chk4 = slim.conv2d_transpose(chk4, num_outputs=256, kernel_size=[3,3], stride=[2,2], padding="VALID") # 3, 5
        chk4 = tf.slice(chk4, [0, 1, 1, 0], [-1, 2, 4, -1]) # 2, 4
        chk4 = batch_norm(chk4)
        chk3 = tf.concat([chk3, chk4], 3) # 2, 4
        chk3 = slim.conv2d_transpose(chk3, num_outputs=512, kernel_size=[3,3], stride=[2,2], padding="VALID") # 5, 9
        chk3 = tf.slice(chk3, [0, 1, 1, 0], [-1, 3, 8, -1]) # 3, 8
        chk3 = batch_norm(chk3)
        chk2 = tf.concat([chk2, chk3], 3) # 3, 8
        chk2 = slim.conv2d_transpose(chk2, num_outputs=1024, kernel_size=[3,3], stride=[2,2], padding="VALID") # 7, 17
        chk2 = tf.slice(chk2, [0, 1, 1, 0], [-1, 6, 15, -1]) # 6, 15
        chk2 = batch_norm(chk2)
        chk1 = tf.concat([chk1, chk2], 3)
        chk1 = batch_norm(chk1)

        # C
        net = slim.fully_connected(tf.reshape(chk1, [batch_size * (LEFT_CONTEXT + seq_len), -1]), 1024, activation_fn=tf.nn.relu)
        net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu)
        net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu)
        net = slim.fully_connected(net, 64, activation_fn=tf.nn.relu)
        net = slim.fully_connected(net, 1, activation_fn=None)
        return net    


# # Model
# 
# Let's build the main graph. Code is mostly self-explanatory.
# 
# A few comments:
# 
# 1) PNG images were used as the input only because this was the format for round1 testset. In practice, raw images should be fed directly from the rosbags.
# 
# 2) We define get_initial_state and deep_copy_initial_state functions to be able to preserve the state of our recurrent net between batches. The backpropagation is still truncated by SEQ_LEN.
# 
# 3) The loss is composed of two components. The first is the MSE of the steering angle prediction in the autoregressive setting -- that is exactly what interests us in the test time. The second components, weighted by the term aux_cost_weight, is the sum of MSEs for all outputs both in autoregressive and ground truth settings. 
# 
# Note: if the saver definition doesn't work for you please make sure you are using tensorflow 0.12rc0 or newer.

# In[ ]:

graph = tf.Graph()

with graph.as_default():
    # inputs  
    # leraning_rate_default : 1e-4
    learning_rate = tf.placeholder_with_default(input=1e-4, shape=())
    keep_prob = tf.placeholder_with_default(input=1.0, shape=())
    aux_cost_weight = tf.placeholder_with_default(input=0.1, shape=())
    
    inputs = tf.placeholder(shape=(BATCH_SIZE,(LEFT_CONTEXT+SEQ_LEN)), dtype=tf.string) # pathes to png files from the central camera
    targets = tf.placeholder(shape=(BATCH_SIZE,(LEFT_CONTEXT+SEQ_LEN),OUTPUT_DIM), dtype=tf.float32) # seq_len x batch_size x OUTPUT_DIM
    targets_normalized = (targets - mean) / std
    targets_normalized = tf.reshape(targets_normalized, [BATCH_SIZE*(SEQ_LEN+LEFT_CONTEXT), 1])
    
    input_images = tf.stack([tf.image.resize_images(tf.image.decode_png(tf.read_file(x)), [HEIGHT, WIDTH]) for x in tf.unstack(tf.reshape(inputs, shape=[(LEFT_CONTEXT+SEQ_LEN) * BATCH_SIZE]))])
#    input_images_norm = tf.cast(input_images, tf.float32) - tf.Variable([123.68, 116.779, 103.939])
    input_images_norm = -1.0 + 2.0 * tf.cast(input_images, tf.float32) / 255.0
    input_images_norm.set_shape([(LEFT_CONTEXT+SEQ_LEN) * BATCH_SIZE, HEIGHT, WIDTH, CHANNELS])
    #pdb.set_trace()
    visual_conditions_reshaped= apply_vision_simple(image=input_images_norm, keep_prob=keep_prob, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    visual_conditions = tf.reshape(visual_conditions_reshaped, [BATCH_SIZE*(SEQ_LEN+LEFT_CONTEXT), 1])
    mse_autoregressive_steering = tf.reduce_mean(tf.squared_difference(visual_conditions, targets_normalized))
    steering_predictions = (visual_conditions * std[0]) + mean[0]
    
    total_loss = mse_autoregressive_steering 
    
    optimizer = get_optimizer(total_loss, learning_rate)

    # tf.scalar_summary
    tf.summary.scalar("MAIN TRAIN METRIC: rmse_autoregressive_steering", tf.sqrt(mse_autoregressive_steering))
    
    # tf.merge_all_summaries
    summaries = tf.summary.merge_all()
    # tf.train.SummaryWriter
    train_writer = tf.summary.FileWriter("chk/basic/AB'C+BN/train_summary", graph=graph)
    valid_writer = tf.summary.FileWriter("chk/basic/AB'C+BN/valid_summary", graph=graph)
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
   


# # Training
# 
# At this point we can start the training procedure.
# 
# We will perform optimization for 100 epochs, doing validation after each epoch. We will keep the model's version that obtains the best performance in terms of the primary loss (autoregressive steering MSE) on the validation set.
# An aggressive regularization is used (`keep_prob=0.25` for dropout), and the validation loss is highly non-monotonical.
# 
# For each version of the model that beats the previous best validation score we will overwrite the checkpoint file and obtain predictions for the challenge test set.

# In[ ]:

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
gpu_options = tf.GPUOptions(allow_growth =True)

checkpoint_dir = os.getcwd() + "/chk/basic/AB'C+BN"
checkpoint_dir_pre = os.getcwd() + "/chk/basic/5*1e-4"

global_train_step = 0
global_valid_step = 0

KEEP_PROB_TRAIN = 0.5

def flatten_list(lst):
	return sum( ([x] if not isinstance(x, list) else flatten(x) for x in lst), [] )

def do_epoch(session, sequences, mode):
    global global_train_step, global_valid_step
    test_predictions = {}
    valid_predictions = {}
    batch_generator = BatchGenerator(sequence=sequences, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
    total_num_steps = 1 + (batch_generator.indices[1] - 1) / SEQ_LEN
    state_gt_cur, controller_final_state_autoregressive_cur = None, None
    acc_loss = np.float128(0.0)
    for step in range(total_num_steps):
        feed_inputs, feed_targets = batch_generator.next()
        feed_dict = {inputs : feed_inputs, targets : feed_targets}
        #pdb.set_trace()
        if controller_final_state_autoregressive_cur is not None:
            feed_dict.update({controller_initial_state_autoregressive : controller_final_state_autoregressive_cur})
        if mode == "train":
            feed_dict.update({keep_prob : KEEP_PROB_TRAIN})
            summary, _, loss, model_predictions = session.run([summaries, optimizer, mse_autoregressive_steering, steering_predictions],
                           feed_dict = feed_dict)
            train_writer.add_summary(summary, global_train_step)
            steering_targets = feed_targets[:, :, 0].flatten()
            model_predictions = model_predictions.flatten()
            stats = np.stack([steering_targets, model_predictions, (steering_targets - model_predictions)**2])
            feed_inputs = feed_inputs[:, LEFT_CONTEXT:].flatten()
            for i, img in enumerate(feed_inputs):
                valid_predictions[img] = stats[:, i]
            global_train_step += 1
        elif mode == "valid":
            model_predictions, summary, loss = session.run([steering_predictions, summaries, mse_autoregressive_steering],
                           feed_dict = feed_dict)
            valid_writer.add_summary(summary, global_valid_step)
            global_valid_step += 1  
            feed_inputs = feed_inputs[:, LEFT_CONTEXT:].flatten()
            steering_targets = feed_targets[:, :, 0].flatten()
            model_predictions = model_predictions.flatten()
            stats = np.stack([steering_targets, model_predictions, (steering_targets - model_predictions)**2])
            for i, img in enumerate(feed_inputs):
                valid_predictions[img] = stats[:, i]
        elif mode == "test":
            #pdb.set_trace()
            #start_t = time.time()
            model_predictions = session.run([steering_predictions], feed_dict = feed_dict) 
            #finish_t = time.time()
            #print "fps = ", finish_t - start_t
            #pdb.set_trace()
            model_predictions = model_predictions[0].flatten()
            feed_inputs = feed_inputs[:, LEFT_CONTEXT:].flatten()
            for i, img in enumerate(feed_inputs):
                test_predictions[img] = model_predictions[i]
        if mode != "test":
            acc_loss += loss
            print '\r', step + 1, "/", total_num_steps, np.sqrt(acc_loss / (step+1)),
    print
    return (np.sqrt(acc_loss / total_num_steps), valid_predictions) if mode != "test" else (None, test_predictions)
    

NUM_EPOCHS=500

best_validation_score = None
with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as session:
    session.run(tf.initialize_all_variables())
    print 'Initialized'
    ckpt_pre = tf.train.latest_checkpoint(checkpoint_dir_pre)
    if ckpt_pre:
        print "Restoring from", ckpt_pre
        variables = tf.trainable_variables()
        variables_to_restore = variables[:32]
        pretrain_saver = tf.train.Saver(variables_to_restore)
        pretrain_saver.restore(sess=session, save_path=ckpt_pre)
        variables = tf.trainable_variables()

    for epoch in range(NUM_EPOCHS):
        print "Starting epoch %d" % epoch
        print "Validation:"
        valid_score, valid_predictions = do_epoch(session=session, sequences=valid_seq, mode="valid")
#        best_validation_score = 100000000000000
        if best_validation_score is None: 
            best_validation_score = valid_score
        if valid_score < best_validation_score:
            saver.save(session, "chk/basic/AB'C+BN/checkpoint-sdc-ch2")
            best_validation_score = valid_score
            print '\r', "SAVED at epoch %d" % epoch,
            with open("chk/basic/AB'C+BN/valid-predictions-epoch%d" % epoch, "w") as out:
                result = np.float128(0.0)
                for img, stats in valid_predictions.items():
                    print >> out, img, stats
                    result += stats[-1]
                    #pdb.set_trace()
            print "Validation unnormalized RMSE:", np.sqrt(result / len(valid_predictions))
            with open("chk/basic/AB'C+BN/test-predictions-epoch%d" % epoch, "w") as out:
                _, test_predictions = do_epoch(session=session, sequences=test_seq, mode="test")
                print >> out, "frame_id,steering_angle"
                result = np.float(0.0)
                for img, pred in test_predictions.items():
                    img = img.replace("challenge_2/Test-final/center/", "")
                    print >> out, "%s,%f" % (img, pred)
#            params = tf.trainable_variables()
#            f = open("./tensorflow_parameter/BASIC_5*10^-4/PRETRAININGWEIGHT_BASIC_5*10^-4_epoch%d.txt" %epoch,'w')
#            print "Save the weight & bias of epoch%d model" %epoch
#            for i in range(23):
#                pickle.dump(session.run(params[i]), f)
#            f.close()
        if epoch != NUM_EPOCHS - 1:
            print "Training"
            _, train_predictions = do_epoch(session=session, sequences=train_seq, mode="train")
            result = np.float128(0.0)
            for img, stats in train_predictions.items():
                result += stats[-1]
            print "Training unnormalized RMSE:", np.sqrt(result / len(train_predictions))


# Basically that's it.
# 
# The model can be further fine-tuned for the challenge purposes by subsetting the training set and setting the aux_cost_weight to zero. It improves the result slightly, but the improvement is marginal (doesn't affect the challenge ranking). For real-life usage it would be probably harmful because of the risk of overfitting to the dev- or even testset.
# 
# Of course, speaking of realistic models, we don't need to constrain our input only to the central camera -- other cameras and sensors can dramatically improve the performance. Also it is useful to think of a realistic delay for the target sequence to make an actual non-zero-latency control possible.
# 
# If something in this writeup is unclear, please write me a e-mail so that I can add the necessary comments/clarifications.
