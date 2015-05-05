
# coding: utf-8

# This is the python-based interface for our paper:
# 
# >**Holistically-Nested Edge Detection**,
# Saining Xie and Zhuowen Tu, http://arxiv.org/abs/1504.06375, 2015
# 

# In[1]:

#import useful util functions

import sys
# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples/hed_release/
sys.path.insert(0, caffe_root + 'python')

import numpy as np
import matplotlib.pyplot as plt
import caffe
import math
import matplotlib.cm as cm
import scipy.misc
import matplotlib.pylab as pylab


get_ipython().magic(u'matplotlib inline')


# In[2]:

#Set up Caffe parameters
def setup_net(model_root, deploy_file, model_file):
    
    net_full_conv = caffe.Net(model_root+deploy_file, model_root+model_file)
    net_full_conv.set_mode_gpu() #Choose between GPU and CPU
    net_full_conv.set_device(3)  #If GPU, Choose an Device ID
    net_full_conv.set_phase_test()
    net_full_conv.set_mean('data', np.load('../../python/caffe/imagenet/ilsvrc_2012_mean.npy'))
    net_full_conv.set_channel_swap('data', (2,1,0))
    net_full_conv.set_raw_scale('data', 255.0)
    return net_full_conv


# In[3]:

#Partition the list into batches
def partition(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]


# In[4]:

#Main testing functions
def batch_testing(batches, model_root, deploy_file, model_file):
    out_lst = []
    for i in range(0, len(batches)):
        if len(batches) >= 5 and i%(len(batches)/5) == 0:
            print model_file.split('.')[0], ' batch ', i, '/', len(batches), ' done...'
        
        im_lst = batches[i]
        batch_size = len(im_lst)
        re_size = im_lst[0].shape
        get_ipython().system(u'cd {model_root} && sed -i "4s/.*/input_dim: "{str(batch_size)}"/" {deploy_file}')
        get_ipython().system(u'cd {model_root} && sed -i "6s/.*/input_dim: "{str(re_size[0])}"/" {deploy_file}')
        get_ipython().system(u'cd {model_root} && sed -i "7s/.*/input_dim: "{str(re_size[1])}"/" {deploy_file}')
        
        net_full_conv = setup_net(model_root, deploy_file, model_file)
        
        caffe_input = np.asarray([net_full_conv.preprocess('data', in_) for in_ in im_lst])
        out = net_full_conv.forward_all(data = caffe_input)
        
        for j in range(0, len(im_lst)):
            temp_dict = {}
            temp_dict['sigmoid-upscore-fuse'] =  out['sigmoid-upscore-fuse'][j, 0, :, :]
            temp_dict['sigmoid-upscore-dsn1'] =  out['sigmoid-upscore-dsn1'][j, 0, :, :]
            temp_dict['sigmoid-upscore-dsn2'] =  out['sigmoid-upscore-dsn2'][j, 0, :, :]
            temp_dict['sigmoid-upscore-dsn3'] =  out['sigmoid-upscore-dsn3'][j, 0, :, :]
            temp_dict['sigmoid-upscore-dsn4'] =  out['sigmoid-upscore-dsn4'][j, 0, :, :]
            temp_dict['sigmoid-upscore-dsn5'] =  out['sigmoid-upscore-dsn5'][j, 0, :, :]
            out_lst.append(temp_dict)
        
    return out_lst


# In[5]:

#Visualization
def plot_single_scale(scale_lst, size):
    pylab.rcParams['figure.figsize'] = size, size/2
    
    plt.figure()
    for i in range(0, len(scale_lst)):
        s=plt.subplot(1,5,i+1)
        plt.imshow(1-scale_lst[i], cmap = cm.Greys_r)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
    plt.tight_layout()


# In[6]:

#Setting test set list
test_lst = ['./peppers_trees.png']


# Test the model on input image, and visualize the results
# 
# two rows
# 
# >top left: HED weighted-fusion layer output, top right: HED average 1-5 output;
# 
# >bottom: output results from each of the five scales
# 
# Note this ipython notebook interface is created for demonstration and visualization. To get full speed during testing, you need to increase the testing batch size, or try to resize the input image, or directly use the Caffe C++ interface

# In[7]:

im = []
for i in range(0, len(test_lst)):
    im.append(caffe.io.load_image(test_lst[i]))
    
batched_im = partition(im, 1)

#Load the pretrained model and deploy prototxt correspondingly
model_root = './models/'
deploy_file = 'hed.prototxt'

model_file = 'hed_bsds.caffemodel'
hed_res = batch_testing(batched_im, model_root, deploy_file, model_file)

pid = 0 #Change this if testing in batch mode
fuse = hed_res[pid]['sigmoid-upscore-fuse']
s1 = hed_res[pid]['sigmoid-upscore-dsn1']
s2 = hed_res[pid]['sigmoid-upscore-dsn2']
s3 = hed_res[pid]['sigmoid-upscore-dsn3']
s4 = hed_res[pid]['sigmoid-upscore-dsn4']
s5 = hed_res[pid]['sigmoid-upscore-dsn5']
ave = (s1 + s2 + s3 + s4 + s5)/5

scale_lst = [fuse, ave]
plot_single_scale(scale_lst, 22)

scale_lst = [s1, s2, s3, s4, s5]
plot_single_scale(scale_lst, 10)

