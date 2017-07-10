#!/usr/bin/env bash

SRC_FOLDER=$1
OUT_FOLDER=$2
NUM_WORKER=1

echo "Extracting optical flow from videos in folder: ${SRC_FOLDER}"
echo "python tools/build_of.py ${SRC_FOLDER} ${OUT_FOLDER} --ext mp4 --num_worker ${NUM_WORKER} --new_width 340 --new_height 256 2>local/errors.log"
python tools/build_of.py ${SRC_FOLDER} ${OUT_FOLDER} --ext mp4 --num_worker ${NUM_WORKER} --new_width 340 --new_height 256 2>local/errors.log
