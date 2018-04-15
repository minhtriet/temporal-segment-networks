import scipy.io
import pdb
import sys
sys.path.append('.')
from pyActionRecog.action_caffe import CaffeNet
import os
import cv2
import numpy as np

from pyActionRecog.utils.video_funcs import default_aggregation_func

flow_x_prefix = 'flow_x_'
flow_y_prefix = 'flow_y_'
rgb_prefix = 'image_'

final_path = "/media/data/mtriet/scnn/experiments/huawei_c3d1.0/final/%s/" % sys.argv[1]

def softmax(raw_score, T=1):
    exp_s = np.exp((raw_score - raw_score.max(axis=-1)[..., None])*T)
    sum_s = exp_s.sum(axis=-1)
    return exp_s / sum_s[..., None]

def build_net(net_proto, net_weights):
    global net
    net = CaffeNet(net_proto, net_weights, 0)

def eval_video(sport, modality, prepad = 0, afterpad=0):
    global net
    seg_swin = scipy.io.loadmat('/media/data/mtriet/combine/seg_swin.m')['seg_swin']
    res = scipy.io.loadmat('/media/data/mtriet/combine/seg_swin.m')['res']
    videoname = scipy.io.loadmat('/media/data/mtriet/combine/seg_swin.m')['videoname'][0]
    video_frame_path = '/media/data/mtriet/dataset/%s_flow_val/%s' % (sport, videoname)
    score_name = 'fc-huawei'

    seg_swin_tsn = np.copy(seg_swin)
    stack_depth = 0
    if modality == 'rgb':
        stack_depth = 1
    elif modality == 'flow':
        stack_depth = 5 
    else:
        raise ValueError()
 
    for segment_index, segment in enumerate(seg_swin):
      frame_cnt = int(segment[3]-segment[2])   # lenght of segment
      
      if frame_cnt <= (prepad + afterpad):
        continue

      num_frame_per_video = len(res) 
      step = (frame_cnt - stack_depth) / (num_frame_per_video-1)
      if step > 0:
          frame_ticks = range(1, min((2 + step * (num_frame_per_video-1)), frame_cnt+1), step)
      else:
          frame_ticks = [1] * num_frame_per_video

      assert(len(frame_ticks) == num_frame_per_video)

      frame_scores = []
      for tick in frame_ticks:
          if modality == 'rgb':
              name = '{}{:05d}.jpg'.format(rgb_prefix, tick)
              frame = cv2.imread(os.path.join(video_frame_path, name), cv2.IMREAD_COLOR)
              scores = net.predict_single_frame([frame,], score_name, frame_size=(340, 256))
              frame_scores.append(scores)
          if modality == 'flow':
              frame_idx = [min(frame_cnt, tick+offset) for offset in xrange(stack_depth)]
              flow_stack = []
              for idx in frame_idx:
                  x_name = '{}{:05d}.jpg'.format(flow_x_prefix, idx)
                  y_name = '{}{:05d}.jpg'.format(flow_y_prefix, idx)
                  flow_stack.append(cv2.imread(os.path.join(video_frame_path, x_name), cv2.IMREAD_GRAYSCALE))
                  flow_stack.append(cv2.imread(os.path.join(video_frame_path, y_name), cv2.IMREAD_GRAYSCALE))
              scores = net.predict_single_flow_stack(flow_stack, score_name, frame_size=(340, 256))
              frame_scores.append(scores)

    print 'video {} done'.format(videoname)
    sys.stdin.flush()
    final = {'seg_swin': seg_swin_tsn, 'res': res}  
    scipy.io.savemat('%s/seg_swin.m' % final_path, final, appendmat=False)
    return np.array(frame_scores)

print("OPERATING PAD 2 SECONDS")
sport = sys.argv[1]
for modality in ['flow', 'rgb']:
  net_weights = "/media/data/mtriet/temporal-segment-networks/models/huawei_%s/%s_%s.caffemodel" % (sport, sport, modality)
  net_proto = "/media/data/mtriet/temporal-segment-networks/models/huawei_%s/tsn_bn_inception_%s_deploy.prototxt" % (sport, modality) 
  build_net(net_proto, net_weights)
  video_scores = eval_video(sport, modality, 2, 2)
  video_pred = np.argmax(default_aggregation_func(video_scores))
