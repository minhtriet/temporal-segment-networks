import argparse
import os
import sys
import math
sys.path.append('.')
import cv2
import numpy as np
import multiprocessing
from sklearn.metrics import confusion_matrix
from pyActionRecog import parse_directory

import pdb
import json

import operator

from pyActionRecog import parse_split_file

import subprocess

def softmax(raw_score, T=1):
    exp_s = np.exp((raw_score - raw_score.max(axis=-1)[..., None])*T)
    sum_s = exp_s.sum(axis=-1)
    return exp_s / sum_s[..., None]


parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'huawei_fb', 'huawei_bb'])
parser.add_argument('split', type=int, choices=[1, 2, 3],
                    help='on which split to test the network')
parser.add_argument('modality', type=str, choices=['rgb', 'flow'])
parser.add_argument('frame_path', type=str, help="root directory holding the frames")
parser.add_argument('net_proto', type=str)
parser.add_argument('net_weights', type=str)
parser.add_argument('--rgb_prefix', type=str, help="prefix of RGB frames", default='image_')
parser.add_argument('--flow_x_prefix', type=str, help="prefix of x direction flow images", default='flow_x_')
parser.add_argument('--flow_y_prefix', type=str, help="prefix of y direction flow images", default='flow_y_')
parser.add_argument('--save_scores', type=str, default=None, help='the filename to save the scores in')
parser.add_argument('--num_worker', type=int, default=1)
parser.add_argument("--caffe_path", type=str, default='./lib/caffe-action/', help='path to the caffe toolbox')
parser.add_argument("--gpus", type=int, nargs='+', default=None, help='specify list of gpu to use')
args = parser.parse_args()
print args

sys.path.append(os.path.join(args.caffe_path, 'python'))
from pyActionRecog.action_caffe import CaffeNet

# build neccessary information
print args.dataset
# split_tp = parse_split_file(args.dataset)

def line2rec(line):
    items = line.split(' ')
    vid = items[0]
    return vid, int(items[1]) # length

if args.dataset == 'huawei_fb':
    split_tp = [line2rec(x) for x in open('data/huawei_splits/test_long_fb.txt')]
else:
    split_tp = [line2rec(x) for x in open('data/huawei_splits/test_long_bb.txt')]

f_info = parse_directory(args.frame_path, args.rgb_prefix, args.flow_x_prefix, args.flow_y_prefix)
    
gpu_list = args.gpus
eval_video_list = split_tp
score_name = 'fc-action'
#score_name = 'fc-huawei'

def build_net():
    global net
    my_id = multiprocessing.current_process()._identity[0] \
        if args.num_worker > 1 else 1
    if gpu_list is None:
        net = CaffeNet(args.net_proto, args.net_weights, my_id-1)
    else:
        net = CaffeNet(args.net_proto, args.net_weights, gpu_list[my_id - 1])

def default_aggregation_func(x):
    return softmax(np.mean(x, axis=0))

def eval_video(video):
    global net
    vid = os.path.basename(video[0])
    num_frame_per_video = video[1]
    print('VIDEO: %s' % vid)
    pdb.set_trace()
    video_frame_path = f_info[0][vid]
    if args.modality == 'rgb':
        cnt_indexer = 1
    elif args.modality == 'flow':
        cnt_indexer = 2
    else:
        raise ValueError(args.modality)
    frame_cnt = f_info[cnt_indexer][vid]

    stack_depth = 0
    if args.modality == 'rgb':
        stack_depth = 1
    elif args.modality == 'flow':
        stack_depth = 5

    step = (frame_cnt - stack_depth) / (num_frame_per_video-1)
    if step > 0:
        frame_ticks = range(1, min((2 + step * (num_frame_per_video-1)), frame_cnt+1), step)
    else:
        frame_ticks = [1] * num_frame_per_video
    pdb.set_trace()
    assert(len(frame_ticks) == num_frame_per_video)
    frame_scores = []
    name = os.path.join(video_frame_path, "%s.aqt" % vid)
    for i, tick in enumerate(frame_ticks):
      if args.modality == 'rgb':
          name = '{}{:05d}.jpg'.format(args.rgb_prefix, tick)
          frame = cv2.imread(os.path.join(video_frame_path, name), cv2.IMREAD_COLOR)
          scores = net.predict_single_frame([frame,], score_name, frame_size=(340, 256))
          frame_scores.append(scores)
      if args.modality == 'flow':
          frame_idx = [min(frame_cnt, tick+offset) for offset in xrange(stack_depth)]
          flow_stack = []
          for idx in frame_idx:
              x_name = '{}{:05d}.jpg'.format(args.flow_x_prefix, idx)
              y_name = '{}{:05d}.jpg'.format(args.flow_y_prefix, idx)
              flow_stack.append(cv2.imread(os.path.join(video_frame_path, x_name), cv2.IMREAD_GRAYSCALE))
              flow_stack.append(cv2.imread(os.path.join(video_frame_path, y_name), cv2.IMREAD_GRAYSCALE))
          scores = net.predict_single_flow_stack(flow_stack, score_name, frame_size=(340, 256))
          frame_scores.append(scores)
#          frame_scores.extend(np.argmax(scores, axis=1))
          video_pred = [np.argmax(default_aggregation_func(x[0])) for x in frame_scores] 
      if (i % 1000) == 0:
        print("evaluated frame %s: " % i)
    np.savetxt("%s.csv" % vid, video_pred, delimiter=',')
    print 'video {} done'.format(vid)
    sys.stdin.flush()
    return np.array(frame_scores)

#build_net()
video_scores = map(eval_video, eval_video_list)
