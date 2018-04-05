import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import sys
import pdb
import json
from matplotlib.patches import Rectangle
import glob
import os

MAP_PATH = "/media/data/mtriet/dataset/script"
GROUND_TRUTH_PATH = "/media/data/mtriet/scnn/experiments/huawei_c3d1.0"
VIDEO_PATH = "/media/data/mtriet/raw_video/%s_eval/" % sys.argv[1]

def plot(video, axe_gt, axe_pred):
  print 'Plotting video %s' % video
  video = video.split('.')[0]
  gt = scipy.io.loadmat('%s/ground_truth/%s/%s' % (GROUND_TRUTH_PATH, sys.argv[1], video))
  y = gt['gt']
  x = np.arange(0, 1.2, 1)
  y = [i[0] for i in y[0]]
  axe_gt.set_ylim([0, 1])
  axe_gt.set_xlim([0, len(y)])
  for i in range(len(y)-1):
    try:
      axe_gt.fill_betweenx(x, i, i+1, color=color_map[y[i]])
    except:
      pdb.set_trace()
  axe_gt.set_xlabel('Ground truth')

  axe_pred.set_ylim([0, 1])
  axe_pred.set_xlim([0, len(y)])

  f = open('../%s.csv' % video, 'r')
  y = f.readlines()
      
  for i in range(len(y)-1):
    klass = int(float(y[i]))
    axe_pred.fill_betweenx(x, i, i+1, color=color_map[ sport_map[klass] ])  
  axe_pred.set_xlabel('Prediction')
  axe_gt.set_yticklabels([])
  axe_pred.set_yticklabels([])

if sys.argv[1] == 'bb':
  with open("%s/bb_classes.json" % MAP_PATH) as data_file:
    sport_map = json.load(data_file)
    sport_map = dict((v,k) for k,v in sport_map.iteritems())
  with open("%s/bb_color.json" % MAP_PATH) as data_file:
    color_map = json.load(data_file)
elif sys.argv[1] == 'fb':
  with open("%s/fb_classes.json" % MAP_PATH) as data_file:
    sport_map = json.load(data_file)
    sport_map = dict((v,k) for k,v in sport_map.iteritems())
  with open("%s/fb_color.json" % MAP_PATH) as data_file:
    color_map = json.load(data_file)
else:
  print "draw_graph video_without_extension <fb/bb>"
  sys.exit(0)


videos = glob.glob('%s/*.mp4' % (VIDEO_PATH))
fig, axes = plt.subplots(len(videos) * 2)
fig.tight_layout()

rects_color = []
for klass in color_map:
  rects_color.append(Rectangle((0, 0), 1, 1, fc=color_map[klass]))
rects_class = [v for k,v in sport_map.iteritems()] 

# load threshold
plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
fig.legend(rects_color, rects_class)

for i, video in enumerate(videos):
  plot(os.path.basename(video), axes[2*i], axes[2*i+1])
  
plt.savefig('fig.png')
