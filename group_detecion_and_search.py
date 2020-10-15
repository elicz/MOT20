import numpy as np
from collections import defaultdict
import math
import random
import os
import cv2
from pathlib import Path
import statistics
import itertools


# 0-lFoot 1-lKnee 2-lHip 3-rHip 4-rKnee 5-rFoot 6-root 7-thorax 8-neck 9-head 10-lHand 11-lElbow 12-lShoulder 13-rShoulder 14-rElbow 15-rHand
JOINT_VIS_ORDER = [6,3,4,5,2,1,0,7,8,9,15,14,13,12,11,10]
BONES = [[0,1],[1,2],[2,3],[3,4],[4,5],[2,6],[3,6],[6,7],[7,8],[8,9],[7,12],[7,13],[10,11],[11,12],[13,14],[14,15]]



# working directory
WORK_SPACE_PATH = '/home/xelias3/deep-high-resolution-net.pytorch/'

# detections dir with tracklets
TRACKLET_DIR= WORK_SPACE_PATH+'data/mot17/tracklets/elias/sdp-ICPR/skeleton/sdp-IDs-dynamic/'

#image dir
IMAGE_DIR = WORK_SPACE_PATH + 'data/mot17/image-sequences/'


# visualization output dir
VISUALIZATION_DIR = WORK_SPACE_PATH + 'data/mot17/vis-ICPR/couples/'


SEQUENCES_TRAIN = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]
SEQUENCES_TEST = ["MOT17-01", "MOT17-03", "MOT17-06", "MOT17-07", "MOT17-08", "MOT17-12", "MOT17-14"]
#SEQUENCES_ALL =  ["MOT17-01", "MOT17-02", "MOT17-03", "MOT17-04", "MOT17-05", "MOT17-06", "MOT17-07", "MOT17-08", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-12", "MOT17-13", "MOT17-14"]
#SEQUENCES_ALL =  ["MOT17-05", "MOT17-06", "MOT17-07", "MOT17-08", "MOT17-09", "MOT17-12"]
SEQUENCES_ALL =  ["MOT17-03"]

def get_start_end_frame(t1):
    start = -1
    end = -1

    for f in t1:
        frame = f[0]
        if(frame > end):
            end = frame
        if((frame < start) or (start == -1)):
            start = frame

    return start, end

def overlaps(a, b):
    """
    Return the amount of overlap, in bp
    between a and b.
    If >0, the number of bp of overlap
    If 0,  they are book-ended.
    If <0, the distance in bp between them
    """

    return min(a[1], b[1]) - max(a[0], b[0])


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

def L2_dist(joint1, joint2):
    dif_x = joint1[0]-joint2[0]
    dif_y = joint1[1]-joint2[1]
    return math.sqrt((dif_x**2) + (dif_y**2))
    

def get_L2_dist_joints(joints1, joints2):
    d = 0
    for j1 in joints1:
        for j2 in joints2:
            d += L2_dist(j1,j2)
    return d

def get_L2_dist_joints_bbox_weighted(joints1, joints2, weight):
    return get_L2_dist_joints(joints1, joints2) / weight


def load_data(track_file):
    dict = {}
    by_ID = {}
    with open(track_file, "r") as file:
        for line in file:
            meta = line.split("#")
            attr = meta[0].split(",")
            frame = int(attr[0])
            id = int(attr[1])
            bb_x = round(float(attr[2]))
            bb_y = round(float(attr[3]))
            w = round(float(attr[4]))
            h = round(float(attr[5]))
            joints = meta[1].split(", ")
            j = []
            for i in range (0, len(joints)-1, 2):
                j1 = joints[i].replace("[", "").replace("]", "").replace("\n", "")
                j2 = joints[i+1].replace("[", "").replace("]", "").replace("\n", "")
                j.append([float(j1), float(j2)])
               
            if(frame in dict):
                record = dict[frame]
            else:
                record = []

            if(id in by_ID):
                r = by_ID[id]
            else:
                r = []

            record.append([id, bb_x, bb_y, w, h, j])
            r.append([frame, bb_x, bb_y, w, h, j])
            dict[frame] = record
            by_ID[id] = r

    for frame in dict:
        t = dict[frame]
        t = sorted(t, key=lambda x: x[0])
        dict[frame] = t

    for id in by_ID:
        t = by_ID[id]
        t = sorted(t, key=lambda x: x[0])
        by_ID[id] = t


    return dict, by_ID



def search_child(tracklets):
    result = {}
    for frame in tracklets:
        detections = tracklets[frame]
        for d1 in detections:
            d1_id, d1_bb_x, d1_bb_y, d1_w, d1_h, joints = d1

            # CHECK CONDITION HERE
            if(len(joints) == 0):
                continue
            h_size = L2_dist(joints[8], joints[9])
            b_size = L2_dist(joints[6], joints[7]) + L2_dist(joints[7], joints[8])
            h2b_ratio = h_size/b_size
            if(h2b_ratio > h2b_limit):
                key = str(d1_id) + "_" + str(d1_id)
                if(key in result):
                    match_frames = result[key]
                else: 
                    match_frames = []

                match_frames.append(frame)
                result[key] = match_frames


    return result


# 10-lHand 13-rHand
def hand_dist(d1, d2):
    dist = -1
    height_ratio = min(d1[4]/d2[4], d2[4]/d1[4])
    # for couples if(height_ratio > 0.6 and len(d1[5])>0 and len(d2[5])>0):  
    if(len(d1[5])>0 and len(d2[5])>0):  

        d1_joints = d1[5]
        d2_joints = d2[5]
     
        d1_lHand = d1_joints[10]
        d1_rHand = d1_joints[15]
        d2_lHand = d2_joints[10]
        d2_rHand = d2_joints[15]

        heights = (d1[4] + d2[4]) / 2
        bone_sum = 0
        for bone in BONES:
            bone_sum += L2_dist(d1_joints[bone[0]], d1_joints[bone[1]])
            bone_sum += L2_dist(d2_joints[bone[0]], d2_joints[bone[1]])

        dist1 = L2_dist(d1_lHand, d2_rHand)
        dist2 = L2_dist(d2_lHand, d1_rHand)

        iou = bb_IoU([d1[1],d1[2],d1[1]+d1[3],d1[2]+d1[4]], [d2[1],d2[2],d2[1]+d2[3],d2[2]+d2[4]])

        dist = min(dist1, dist2)/heights
        #print(dist)

    return dist


def search_hands(tracklets, by_ID, min_length, k):
    result = []

    if(min_length ==0):
        return result


    # for each pair of tracks, returns a tuple, sum of score and number of frames together
    scores = {}
    for frame in tracklets:
        detections = tracklets[frame]
        for d1 in detections:
            d1_id, d1_bb_x, d1_bb_y, d1_w, d1_h, d1_joints = d1
            for d2 in detections:
                d2_id, d2_bb_x, d2_bb_y, d2_w, d2_h, d2_joints = d2
                if(d2_id >= d1_id):
                    continue

                iou = bb_IoU([d1_bb_x, d1_bb_y, d1_bb_x + d1_w, d1_bb_y + d1_h], [d2_bb_x, d2_bb_y, d2_bb_x + d2_w, d2_bb_y + d2_h])


                # CHECK CONDITION HERE
                dist = hand_dist(d1, d2)
                if(dist == -1 or iou < 0.1):
                    continue

                key = str(d1_id) + "_" + str(d2_id)
                if(key in scores):
                    score = scores[key]
                else: 
                    score = [0,0]

                score = [score[0]+dist, score[1]+1]
                scores[key] = score


    # sum divided by mutual frames count
    scores_sorted = []
    for key in scores:
        score_sum, length = scores[key] 
        if(length>=min_length):
            id1, id2 = key.split("_")
            start1,end1 = get_start_end_frame(by_ID[int(id1)])
            start2,end2 = get_start_end_frame(by_ID[int(id2)])
            union = overlaps([start1, end1],[start2,end2])
            intersect = end1-start1 + end2-start2 - union
            score = score_sum/length
            scores_sorted.append([key, score, length])


    scores_sorted = sorted(scores_sorted, key=lambda x: x[1])
    #print(scores_sorted)

    counter = 0
    for item in scores_sorted:
        if(counter<k):
            key, s, l = item
            print("{}: sum {} | len {}".format(key, str(s), str(l)))
            result.append(key)

        counter += 1
   
    return result


def search_children(tracklets, by_ID, min_length, k):
    result = []

    if(min_length ==0):
        return result


    # for each pair of tracks, returns a tuple, sum of score and number of frames together
    scores = {}
    for frame in tracklets:
        detections = tracklets[frame]
        for d1 in detections:
            d1_id, d1_bb_x, d1_bb_y, d1_w, d1_h, d1_joints = d1
            for d2 in detections:
                d2_id, d2_bb_x, d2_bb_y, d2_w, d2_h, d2_joints = d2
                if(d2_id >= d1_id):
                    continue

                if(len(d1_joints) < 1 or len(d2_joints)<1):
                    continue


                # CHECK CONDITION HERE
                dist = L2_dist([d1_bb_x + 0.5*d1_w, d1_bb_y + 0.5*d1_h], [d2_bb_x + 0.5*d2_w, d2_bb_y + 0.5*d2_h])
                h_ratio = d1_h/d2_h
                min_h_ratio = min(h_ratio, d2_h/d1_h)
                iou = bb_IoU([d1_bb_x, d1_bb_y, d1_bb_x + d1_w, d1_bb_y + d1_h], [d2_bb_x, d2_bb_y, d2_bb_x + d2_w, d2_bb_y + d2_h])
                base_dif = abs((d1_bb_y+d1_h)-(d2_bb_y+d2_h))
                smaller_h = min(d1_h, d2_h)
                bigger_h = max(d1_h, d2_h)
                bigger_w = max(d1_w, d2_w)
                d1_head = L2_dist(d1_joints[7], d1_joints[9])
                d1_body = L2_dist(d1_joints[6], d1_joints[7])
                d1_h2b_ratio = d1_head/d1_body
                d2_head = L2_dist(d2_joints[7], d2_joints[9])
                d2_body = L2_dist(d2_joints[6], d2_joints[7])
                d2_h2b_ratio = d2_head/d2_body
                # remove outliers
                if(d1_head<10 or d2_head<10 or d1_body<10 or d2_body<10):
                    continue
                if(d1_joints[7][1]-d1_joints[9][1] < 5 or d2_joints[7][1]-d2_joints[9][1]<5):
                    continue

                proportion_ratio = d1_h2b_ratio/d2_h2b_ratio

                if(dist>(bigger_w) or (base_dif>(0.33*bigger_h)) or (min_h_ratio>0.8)):
                   continue


                key = str(d1_id) + "_" + str(d2_id)
                if(key in scores):
                    a,b,c = scores[key]
                    #print("{}, {}, {}".format(a,b,c,))
                    a.append(d1_h2b_ratio)
                    b.append(d2_h2b_ratio)
                    c += 1
                    #print("{}, {}, {}".format(a,b,c,))

                    score = [a,b,c]
                else: 
                    score = [[d1_h2b_ratio],[d2_h2b_ratio],0]
                
                scores[key] = score


    # sum divided by count and togetherness
    scores_sorted = []
    for key in scores:
        d1_scores, d2_scores, length = scores[key]
        if(length>=min_length):
            m_d1 = statistics.median(d1_scores)
            m_d2 = statistics.median(d2_scores)
            diff = abs(m_d1 - m_d2)
            #score_sum = (h_sum + 2*p_sum)/length
            scores_sorted.append([key, diff, length])

    scores_sorted = sorted(scores_sorted, key=lambda x: x[1], reverse=True)
    print(scores_sorted)

    counter = 0
    for item in scores_sorted:
        if(counter<k):
            key, s, l = item
            print("{}: sum {} | len {}".format(key, str(s), str(l)))
            result.append(key)

        counter += 1
   
    return result


def search_child(tracklets, by_ID, min_length, k):
    result = []
    if(min_length ==0):
        return result
    scores = {}
    for frame in tracklets:
        detections = tracklets[frame]
        for d1 in detections:
            d1_id, d1_bb_x, d1_bb_y, d1_w, d1_h, d1_joints = d1
            if(len(d1_joints) < 1):
                continue
            d1_head = L2_dist(d1_joints[7], d1_joints[8]) + L2_dist(d1_joints[8], d1_joints[9])
            d1_leg = (L2_dist(d1_joints[0], d1_joints[1])+L2_dist(d1_joints[1], d1_joints[2])+L2_dist(d1_joints[3], d1_joints[4])+L2_dist(d1_joints[4], d1_joints[5]))/2
            d1_body = L2_dist(d1_joints[6], d1_joints[7]) + L2_dist(d1_joints[7], d1_joints[8]) + L2_dist(d1_joints[8], d1_joints[9])+ d1_leg
            d1_h2b_ratio = d1_head/d1_body
            # remove outliers
            if(d1_head<5 or d1_body<10 or d1_joints[7][1]-d1_joints[9][1] < 5 or d1_bb_y<0):
                continue
            key = str(d1_id)
            if(key in scores):
                 a,b = scores[key]
                 a.append(d1_h2b_ratio)
                 b += 1
                 score = [a,b]
            else: 
                 score = [[d1_h2b_ratio],0]
            scores[key] = score
    scores_sorted = []
    for key in scores:
        d1_scores, length = scores[key]
        if(length>=min_length):
            m_d1 = statistics.median(d1_scores)
            scores_sorted.append([key, m_d1, length])
    scores_sorted = sorted(scores_sorted, key=lambda x: x[1], reverse=True)
    counter = 0
    for item in scores_sorted:
        if(counter<k):
            key, s, l = item
            #if(s<0.45):
            #    continue
            print("{}: sum {} | len {}".format(key, str(s), str(l)))
            result.append(key)
        counter += 1
    return result


##
# Visulizes images with bounding box overlay
# @ predictionFile - matlab file with prediction
# @ seqID - denotes directory name where images are stored
##
def visualize_groups(tracklets, results, seqID):
    counter = 0
    dict = {}
    write_dir = VISUALIZATION_DIR+seqID
    colors = {}
        
    if not (os.path.exists(write_dir)):
        os.mkdir(write_dir, 0o755);

    for key in results:
        r, g, b = round(255*random.random()) , round(255*random.random()), round(255*random.random())
        colors[key] = (r, g, b)

        ids = [int(s) for s in key.split('_')]


        for frame in tracklets:

            if(frame in dict):
                meta = dict[frame]
            else:
                meta = []


            
            for t in tracklets[frame]:
                if(t[0] in ids):
                    data = [key]
                    for i in range(1,len(t)):
                        data.append(t[i])
                    meta.append(data)
            dict[frame] = meta

    for frame in dict:
        if(len(dict[frame])<1):
            continue

        img_path = IMAGE_DIR + seqID + "/" + str(frame).zfill(6) + ".jpg"
        image = cv2.imread(img_path,cv2.IMREAD_COLOR)

        # DRAW BOXES
        inc = 5
        for meta in dict[frame]:
            image = cv2.rectangle(image, (meta[1]-inc,meta[2]-inc), (meta[1]+meta[3]+inc, meta[2]+meta[4]+inc), (0,255,255), thickness=2, lineType=cv2.LINE_AA, shift=0)

            joints = meta[5]
            if(len(joints)<1):
                continue
            # DRAW JOINTS
            c = 0
            for joint in joints:
                #if(c==10 or c==15):
                if(c==9 or c==7):
                    image = cv2.circle(image, (int(joint[0]),int(joint[1])), 5, (255,0,255) , thickness=1, lineType=cv2.LINE_AA, shift=0)
                
                else:
                    image = cv2.circle(image, (int(joint[0]),int(joint[1])), 5, (0,255,255) , thickness=1, lineType=cv2.LINE_AA, shift=0)
                c = c+1

            # DRAW BONES
            for bone in BONES:
                image = cv2.line(image, (int(joints[bone[0]][0]), int(joints[bone[0]][1])), (int(joints[bone[1]][0]), int(joints[bone[1]][1])), (2,106,253), 1, cv2.LINE_AA)



        frame_nr = Path(img_path).resolve().stem
        cv2.imwrite(VISUALIZATION_DIR+seqID+'/'+frame_nr+'.jpg',image)
        counter +=1
        print(" visualized {} / {}    ".format(counter, len(dict)), end="\r")


T = 0.0225
LIM_LEN = 25
size_limit = 0.5
iou_limit = 0.05
h2b_limit = 0.6


IOU_GROUP_LIMIT = 0.01
HEIGHT_LIMIT = 0.7
GROUP_TIME_LIMIT = 20
GROUP_SIZE = 4
BASE_RATIO = 0.25



# couple search params (5th 0/1?)
k = [0,0,4,1,0,3,1,0,0,0,2,1,0,0]
lim_len = [0,0,50,150,0,25,25,0,0,0,25,25,0,0]


# kid search params
#5-3x (short)
#6-4x /two very short)
#7-2x (one is separated - show as picture
#8-2x 
#9-2x
#12-1x
#k = [4,4,1,2,2,1]
#lim_len = [15,10,20,15,30,10]



counter =0
for sequence in SEQUENCES_ALL:
    print(sequence)
    tracklets, by_ID = load_data(TRACKLET_DIR + sequence + ".txt")
    #result = search_child(tracklets, by_ID, lim_len[counter], k[counter])
    result =  search_hands(tracklets, by_ID, 30, 4)
    visualize_groups(tracklets, result, sequence)
    counter += 1


