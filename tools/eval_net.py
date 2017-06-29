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

from pyActionRecog import parse_directory
from pyActionRecog import parse_split_file

from pyActionRecog.utils.video_funcs import default_aggregation_func

def softmax(x):
    """Compute softmax"""
    mx = np.max(x)
    e_x = np.exp(x - mx)
    return e_x / e_x.sum()

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
    label = items[1]
    vid = items[0]
    return vid, label

split_tp = [line2rec(x) for x in open('data/huawei_splits/test.txt')]

f_info = parse_directory(args.frame_path,
                         args.rgb_prefix, args.flow_x_prefix, args.flow_y_prefix)

gpu_list = args.gpus

#eval_video_list = split_tp[args.split - 1][1]

eval_video_list = split_tp
# score_name = 'fc-action'
score_name = 'fc-huawei'


def build_net():
    global net
    my_id = multiprocessing.current_process()._identity[0] \
        if args.num_worker > 1 else 1
    if gpu_list is None:
        net = CaffeNet(args.net_proto, args.net_weights, my_id-1)
    else:
        net = CaffeNet(args.net_proto, args.net_weights, gpu_list[my_id - 1])


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

video_pred = [np.argmax(default_aggregation_func(x[0])) for x in video_scores]
max_scores = [np.max(default_aggregation_func(x[0])) for x in video_scores]

for index , x in enumerate(max_scores):
    print "%s %s %s" % (x, video_pred[index], eval_video_list[index])

video_pred = [str(x) for x in video_pred]

video_labels = [x[1] for x in video_scores]
cf = confusion_matrix(video_labels, video_pred).astype(float)
print cf
cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit/cls_cnt

print cls_acc

print 'Accuracy {:.02f}%'.format(np.mean(cls_acc)*100)

if args.save_scores is not None:
    np.savez(args.save_scores, scores=video_scores, labels=video_labels)
