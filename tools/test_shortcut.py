import sys
import pdb
import subprocess
if __name__ == '__main__':
    DATASET = sys.argv[1]
    MODALITY = sys.argv[2]
    if len(sys.argv) < 3:
        print "USAGE: test_shortcut dataset modality"
        quit()  
   
    LATEST = subprocess.check_output("ls -t models/%s/*_%s*.caffemodel | head -1" % (DATASET, MODALITY), shell=True).rstrip()
    if DATASET == 'huawei_fb': 
        FRAME_FOLDER="fb_flow"
    else:
        FRAME_FOLDER="bb_flow"

    COMMAND="python tools/eval_net.py %s 1 %s /media/data/mtriet/dataset/%s/ models/%s/tsn_bn_inception_%s_deploy.prototxt %s --save_score score.txt" % (DATASET, MODALITY, FRAME_FOLDER, DATASET, MODALITY, LATEST)

    print COMMAND
    subprocess.call(COMMAND, shell=True)

