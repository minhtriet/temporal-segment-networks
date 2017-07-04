import argparse
import os
import sys
import math
sys.path.append('.')
import cv2
import numpy as np
import multiprocessing
from sklearn.metrics import confusion_matrix

import pdb
import json

import operator

#from pyActionRecog import parse_directory
from pyActionRecog import parse_split_file

import subprocess
from pyActionRecog.utils.video_funcs import default_aggregation_func

def softmax(x):
    """Compute softmax"""
    mx = np.max(x)
    e_x = np.exp(x - mx)
    return e_x / e_x.sum()

def priors(x):

    def load_json(path):
        with open(path) as fj:
            data = json.load(fj)
        return data

    priors = []
    if x == 'huawei_fb':
        classes = load_json('data/huawei_splits/classes_fb.json')
        print "List classes:" 
        print classes
        train_file = 'data/huawei_splits/train_fb.txt'
        for i in classes:
            batcmd="grep '%s' %s | wc -l" % (i, train_file)
            result = subprocess.check_output(batcmd, shell=True)
            priors.append(int(result))

    priors = [-1 if x == 0 else x for x in priors]
    return priors

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
parser.add_argument('--num_frame_per_video', type=int, default=25,
                    help="prefix of y direction flow images")
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
split_tp = parse_split_file(args.dataset)

def line2rec(line):
    items = line.split(' ')
    label = int(items[2])
    vid = items[0]
    return vid, label, int(items[1]) # length

if args.dataset == 'huawei_fb':
    split_tp = [line2rec(x) for x in open('data/huawei_splits/test_fb.txt')]
else:
    split_tp = [line2rec(x) for x in open('data/huawei_splits/test_bb.txt')]

f_info = [{}, {}, {}]

for x in split_tp:
    f_info[0][os.path.basename(x[0])] = x[0]
    f_info[1][os.path.basename(x[0])] = x[2]
    f_info[2][os.path.basename(x[0])] = x[2]
    # assume that length of rgb and flow is same
    
# f_info = parse_directory(args.frame_path,
#                         args.rgb_prefix, args.flow_x_prefix, args.flow_y_prefix, list_file)
gpu_list = args.gpus

#eval_video_list = split_tp[args.split - 1][1]

eval_video_list = split_tp
score_name = 'fc-huawei'


def build_net():
    global net
    my_id = multiprocessing.current_process()._identity[0] \
        if args.num_worker > 1 else 1
    if gpu_list is None:
        net = CaffeNet(args.net_proto, args.net_weights, my_id-1)
    else:
        net = CaffeNet(args.net_proto, args.net_weights, gpu_list[my_id - 1])

def output(video_scores, prior=None):
    if prior != None:
        temp = [default_aggregation_func(x[0]) for x in video_scores]
        temp = [ map(operator.truediv, x, prior) for x in temp]
        video_pred = [ np.argmax(x) for x in temp]
        max_scores = [ np.max(x) for x in video_scores]
    else:
        video_pred = [np.argmax(default_aggregation_func(x[0])) for x in video_scores]
        max_scores = [np.max(default_aggregation_func(x[0])) for x in video_scores]

    for index , x in enumerate(max_scores):
        print "%s %s %s" % (x, video_pred[index], eval_video_list[index])
    pdb.set_trace()
    video_labels = [x[1] for x in video_scores]

    cf = confusion_matrix(video_labels, video_pred).astype(float)
    print cf
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    cls_acc = cls_hit/cls_cnt

    print cls_acc

    print 'Mean accuracy over classes {:.02f}%'.format(np.mean(cls_acc)*100)
    print 'Accuracy over classes: %s' % (np.mean(cls_acc)*100)
    print 'Accuracy over samples: %s'% (cls_hit / np.sum(cf)*100)


def eval_video(video):
    global net
    label = video[1]
    vid = os.path.basename(video[0])
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

    step = (frame_cnt - stack_depth) / (args.num_frame_per_video-1)
    if step > 0:
        frame_ticks = range(1, min((2 + step * (args.num_frame_per_video-1)), frame_cnt+1), step)
    else:
        frame_ticks = [1] * args.num_frame_per_video

    assert(len(frame_ticks) == args.num_frame_per_video)

    frame_scores = []
    for tick in frame_ticks:
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
            frame_scores.append(softmax(scores))

    print 'video {} done'.format(vid)
    sys.stdin.flush()
    return np.array(frame_scores), label

if args.num_worker > 1:
    pool = multiprocessing.Pool(args.num_worker, initializer=build_net)
    video_scores = pool.map(eval_video, eval_video_list)
else:
    build_net()
    video_scores = map(eval_video, eval_video_list)

output(video_scores)

print 'Divide score by prior'
prior = priors(args.dataset)
print "Prior %s " % prior
output(video_scores, prior)

