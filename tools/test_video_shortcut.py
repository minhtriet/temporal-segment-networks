import sys
import pdb
import subprocess
import glob

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "USAGE: test_shortcut dataset modality"
        quit()  
   
    DATASET = sys.argv[1]    
    MODALITY = sys.argv[2]
    LATEST = subprocess.check_output("ls -t models/%s/*_%s*.caffemodel | head -1" % (DATASET, MODALITY), shell=True).rstrip()
    
    ds = DATASET.split('_')[1]
    
    FRAME_FOLDER = '/media/data/mtriet/temporal-segment-networks/tools/frames/'
    VIDEO_FOLDER = '/media/data/mtriet/raw_video/%s_eval/' % ds
    # extract frames
    video_folders = glob.glob("%s*.mp4" % VIDEO_FOLDER)
    for video in video_folders:
      try:
        video = video.split('.')[0]
        os.makedir("%s%s" % (FRAME_FOLDER, video))
      except:
        print "%s: flow and rgb data existed" % video 
        continue
      COMMAND="bash scripts/extract_optical_flow.sh /media/data/mtriet/raw_video/%s_eval/%s.mp4 %s" % (DATASET[-2:], video, FRAME_FOLDER)
      print COMMAND
      subprocess.call(COMMAND, shell=True)
    # create eval list
    with open("data/huawei_splits/test_long_%s.txt" % ds, 'w') as f:
      for folder in glob.glob("%s/*" % FRAME_FOLDER):
        for s in glob.glob(folder):
          num_frame_per_video = len( glob.glob("%s/*.jpg" % s) )/3 
          f.write("%s %d\n" % (folder, num_frame_per_video))
    # eval
#    COMMAND="python tools/eval_video.py %s 1 %s %s models/%s/tsn_bn_inception_%s_deploy.prototxt %s --save_score /media/data/mtriet/temporal-segment-networks/tools/score.txt" % (DATASET, MODALITY, FRAME_FOLDER, DATASET, MODALITY, LATEST)
    PROTOTXT = "/media/data/mtriet/temporal-segment-networks/models/ucf101/tsn_bn_inception_flow_deploy.prototxt"
    LATEST = "/media/data/mtriet/temporal-segment-networks/models/ucf101/ucf101_split_1_tsn_flow_reference_bn_inception.caffemodel"
    COMMAND="python tools/eval_video.py %s 1 %s %s %s %s --save_score /media/data/mtriet/temporal-segment-networks/tools/score.txt" % (DATASET, MODALITY, FRAME_FOLDER, PROTOTXT, LATEST)
    print COMMAND
    subprocess.call(COMMAND, shell=True)
