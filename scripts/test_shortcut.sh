#!/bin/bash

DATASET=$1
MODALITY=$2

if [ "$#" -ne 2 ]; then
    echo "USAGE: test_shortcut dataset modality"
    exit 1
fi
 
# SELECT MODEL
LATEST=`ls -t models/*_${MODALITY}*.caffemodel | head -1`

if [ $DATASET == 'huawei_fb' ]; then
    FRAME_FOLDER="fb_flow"
else
    FRAME_FOLDER="bb_flow"
fi

COMMAND="$(python tools/eval_net.py ${DATASET} 1 $MODALITY /media/data/mtriet/dataset/$FRAME_FOLDER/ models/$DATASET/tsn_bn_inception_${MODALITY}_deploy.prototxt $LATEST --save_score score.txt)"

echo "${COMMAND}"

python tools/eval_net.py $DATASET 1 $MODALITY /media/data/mtriet/dataset/$FRAME_FOLDER/ models/$DATASET/tsn_bn_inception_${MODALITY}_deploy.prototxt $LATEST --save_score score.txt
