import scipy.io
import pdb
import sys
sys.path.append('.')
from pyActionRecog.action_caffe import CaffeNet
import os
import cv2
import numpy as np

flow_x_prefix = 'flow_x_'
flow_y_prefix = 'flow_y_'
rgb_prefix = 'image_'

def softmax(raw_score, T=1):
    exp_s = np.exp((raw_score - raw_score.max(axis=-1)[..., None])*T)
    sum_s = exp_s.sum(axis=-1)
    return exp_s / sum_s[..., None]

def build_net(net_proto, net_weights):
    global net
    net = CaffeNet(net_proto, net_weights, 0)

def eval_video(sport, modality):
    global net
    seg_swin = scipy.io.loadmat('/media/data/mtriet/combine/seg_swin.m')['seg_swin']
    videoname = scipy.io.loadmat('/media/data/mtriet/combine/seg_swin.m')['videoname'][0]
    video_frame_path = '/media/data/mtriet/dataset/%s_flow_val/%s' % (sport, videoname)
    score_name = 'fc-huawei'
    for segment in seg_swin:
      if modality == 'rgb':
          cnt_indexer = 1
      elif modality == 'flow':
          cnt_indexer = 2
      else:
          raise ValueError()
      frame_cnt = int(segment[3]-segment[2])   # lenght of segment
      num_frame_per_video = int(segment[3] - segment[2])    # length of video

      stack_depth = 0
      if modality == 'rgb':
          stack_depth = 1
      elif modality == 'flow':
          #stack_depth = 5
          stack_depth = 10 

      frame_ticks = [1] * num_frame_per_video

      assert(len(frame_ticks) == num_frame_per_video)
      frame_scores = []
      name = os.path.join(video_frame_path, "%s.aqt" % videoname)
      for i, tick in enumerate(frame_ticks):
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
            scores = net.predict_single_flow_stack(flow_stack, score_name, frame_size=(340, 256))
            frame_scores.append(scores)
  #          frame_scores.extend(np.argmax(scores, axis=1))
        video_pred = [np.argmax(softmax(x[0])) for x in frame_scores] 
        if (i % 220) == 0:
          print("evaluated frame %s: " % i)
      np.savetxt("%s.csv" % videoname, video_pred, delimiter=',')
      print 'video {} done'.format(videoname)
      sys.stdin.flush()
      return np.array(frame_scores)

sport = sys.argv[1]
for modality in ['flow', 'rgb']:
  net_weights = "/media/data/mtriet/temporal-segment-networks/models/huawei_%s/%s_%s.caffemodel" % (sport, sport, modality)
  net_proto = "/media/data/mtriet/temporal-segment-networks/models/huawei_%s/tsn_bn_inception_%s_deploy.prototxt" % (sport, modality) 
  build_net(net_proto, net_weights)
  eval_video(sport, modality)
