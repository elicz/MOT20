import numpy as np
from collections import defaultdict
import math
import random
import os
import cv2
from pathlib import Path
import statistics


ALLOWED_TYPES = [1,2,7,8,12]

# working directory
WORK_SPACE_PATH = '/home/xelias3/deep-high-resolution-net.pytorch/'

# detections dir with tracklets
TRACKLET_DIR= WORK_SPACE_PATH+'data/mot17/tracklets/elias/sdp-ICPR/sdp-iou-dynamic-ids/'

#image dir
IMAGE_DIR = WORK_SPACE_PATH + 'data/mot17/image-sequences/'

#GT dir
GT_DIR = WORK_SPACE_PATH + 'data/mot17/meta/ground-truth/'


# visualization output dir
VISUALIZATION_DIR = WORK_SPACE_PATH + 'data/mot17/vis-temp-ICPR/'

SEQUENCES = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]
SEQUENCES = ["MOT17-09"]

def bb_IoU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if(interArea > 0):
        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
        # compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    return iou


def analyzeGT(detection_file, seqID):
    dict = {}
    with open(detection_file, "r") as file:
        for line in file:
            attr = line.split(",")
            frame = int(attr[0])
            id = int(attr[1])
            bb_x = round(float(attr[2]))
            bb_y = round(float(attr[3]))
            w = round(float(attr[4]))
            h = round(float(attr[5]))
            consider = int(attr[6])
            if(consider ==0):
                continue


            if(id in dict):
                meta = dict[id]
            else:
                meta = []

            meta.append([frame, bb_x, bb_y, w, h])
            dict[id] = meta

    for id in dict:
        t = dict[id]
        t = sorted(t, key=lambda x: x[0])
        dict[id] = t

    counter = 0
    ious = []
    for id in dict:
        for f in range(0, len(dict[id])-1):
            current = dict[id][f]
            next = dict[id][f+1]
            iou = bb_IoU([current[1],current[2], current[1]+current[3], current[2]+current[4]], [next[1],next[2],next[1]+next[3],next[2]+next[4]])
            ious.append(iou)

            

    print("{}: min {} | avg {} | max {}".format(seqID, min(ious), statistics.mean(ious), max(ious)))



##
# Visulizes images with bounding box and skeleton overlay
# @ track file - file with tracks in format
# @ seqID - denotes sequences id (same with directory name where images are stored)
##
def visualize_bbox(detection_file, seqID):
    counter = 0
    dict = {}
    write_dir = VISUALIZATION_DIR+seqID
    colors = []
    for i in range (0, 1000):
        r, g, b = round(255*random.random()) , round(255*random.random()), round(255*random.random())
        colors.append((r, g, b))

    if not (os.path.exists(write_dir)):
        os.mkdir(write_dir, 0o755);


    with open(detection_file, "r") as file:
        for line in file:
            attr = line.split(",")
            frame = int(attr[0])
            id = int(attr[1])
            bb_x = round(float(attr[2]))
            bb_y = round(float(attr[3]))
            w = round(float(attr[4]))
            h = round(float(attr[5]))
            if(frame in dict):
                meta = dict[frame]
            else:
                meta = []

            meta.append([id, bb_x, bb_y, w, h])
            dict[frame] = meta

    for frame in dict:
        img_path = IMAGE_DIR + seqID + "/" + str(frame).zfill(6) + ".jpg"
        image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

        # DRAW BOXES
        for meta in dict[frame]:
            image = cv2.rectangle(image, (meta[1],meta[2]), (meta[1]+meta[3],meta[2]+meta[4]), colors[int(meta[0])], thickness=4, lineType=cv2.LINE_AA, shift=0)
        frame_nr = Path(img_path).resolve().stem
        cv2.imwrite(VISUALIZATION_DIR+seqID+'/'+frame_nr+'.jpg',image)

        counter +=1
        print(" visualized {} / {}    ".format(counter, len(dict)), end="\r")


for sequence in SEQUENCES:
    visualize_bbox(TRACKLET_DIR + sequence + ".txt", sequence)
    #analyzeGT(GT_DIR+sequence+".txt", sequence)