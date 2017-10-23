import sys
import pdb
import subprocess
import glob

FRAME_FOLDER = '/media/data/mtriet/temporal-segment-networks/tools/frames/'

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "USAGE: test_shortcut dataset modality"
        quit()  
   
    DATASET = sys.argv[1]    
    MODALITY = sys.argv[2]
    LATEST = subprocess.check_output("ls -t models/%s/*_%s*.caffemodel | head -1" % (DATASET, MODALITY), shell=True).rstrip()
    
    ds = DATASET.split('_')[1]

    COMMAND="bash scripts/extract_optical_flow.sh /media/data/mtriet/raw_video/%s_eval/ %s" % (DATASET[-2:], FRAME_FOLDER)
    print COMMAND
    subprocess.call(COMMAND, shell=True)

    video_folders = glob.glob("%s*", FRAME_FOLDER)
    with open("data/huawei_splits/test_long_%s.txt" % ds, 'w') as f:
      for s in video_folders:        
        frames = len( glob.glob(s) )/3
        num_frame_per_video = int(s.split(' ')[1]) / 3
        print "Length is %d " % num_frame_per_video

    COMMAND="python tools/eval_video.py %s 1 %s /media/data/mtriet/dataset/%s/ models/%s/tsn_bn_inception_%s_deploy.prototxt %s --save_score /media/data/mtriet/temporal-segment-networks/tools/score.txt --num_frame_per_video %s" % (DATASET, MODALITY, FRAME_FOLDER, DATASET, MODALITY, LATEST, num_frame_per_video)
    print COMMAND
    subprocess.call(COMMAND, shell=True)
