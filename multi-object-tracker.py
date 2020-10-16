# 1. IMPORTS
#============
import numpy as np
from collections import defaultdict
import math
import os
from lapsolver import solve_dense
import timeit
import heapq 

# 2. ENVIRONMENT PARAMETERS
#===========================
WORK_SPACE_PATH = '/home/xelias3/deep-high-resolution-net.pytorch/'	        # working directory
DET_DIR= WORK_SPACE_PATH+'data/mot17/detections/sdp/'			            # detections dir with unordered hypotheses
tracker = "sdp-MTAP/baselines/sdp-general-munkres-reid"                     # directory name, where output tracks should be stored
TRACK_OUTPUT_DIR = WORK_SPACE_PATH + "data/mot17/tracks/" + tracker + "/"   # complete path to directory where output tracks should be stored


# MOT17 and MOT20 test and train sequences and allowed types
ALLOWED_TYPES = [1,2,7,8,12]
SEQUENCES_TRAIN_20 = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
SEQUENCES_TRAIN = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]
SEQUENCES_TEST = ["MOT17-01", "MOT17-03", "MOT17-06", "MOT17-08", "MOT17-07", "MOT17-12", "MOT17-14"]
SEQUENCES_TEST_20 = ["MOT20-04", "MOT20-06", "MOT20-07", "MOT20-08"]
SEQUENCES_ALL = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13", "MOT17-01", "MOT17-03", "MOT17-06", "MOT17-07", "MOT17-08", "MOT17-12", "MOT17-14"]

# 3. TRACKING PARAMETERS
#========================
IOU_TRACKING = 0.2 		    # IOU tracking limit to match two detections, default=0.25
SIZE_LIMIT = 3 			    # Minimum number of frames required to constitute a track, default=5
INTERPOLATE = True		    # Interpolate poses in re-identified tracks, default=True
MIN_LENGTH_TO_MATCH = 3	    # Minimum length of track required for matching fragmented tracks, default=3
MATCH_FRAMES = 2		    # Exact number of frames to be projected for matching fragmented tracks, default=2 (event. 3) 
MATCH_BASIS = 30		    # If fragmented track has more frames than MIN_LENGTH_TO_MATCH, maximum number of frames to take into account when projecting (minimum of (length,match_basis is taken), default=30
MATCH_MAX_FRAME_GAP = 50	# Max allowed gap between to-be-matched fragmented tracklets, default=50 
REQUIRED_MATCH_SCORE = 0.3	# Min IOU score to match two fragmented tracks based on MATCH_FRAMES X MATCH_FRAMES sum of IOU, default=0.25 
CACHE = 7			        # Number of frames to keep residual detections in memory for matching


# OPTIMUM FOR TEST SEQUENCES
# LOCAL PARAMETERS DERIVED FROM TRAINING DATA
#TRAIN_IOU_TRACKING = [0.3, 0.5, 0.1, 0.5, 0.1, 0.001, 0.1]
#TRAIN_SIZE_LIMIT = [3, 3, 3, 3, 7, 5, 7]
#TRAIN_MATCH_MAX_FRAME = [50, 100, 100, 100, 25, 50, 50]
#TRAIN_MATCH_IOU = [0.2, 0.2, 0.4, 0.1, 0.4, 0.1, 0.4]
#TRAIN_CACHE_SIZE = [10, 10, 3, 5, 3, 10, 3]



# 4. I/O METHODS
#================

# LOAD DETECTIONS
def get_hypotheses(hypo_file):
    HYPO = {}
    with open(hypo_file) as f_hypo:
        poses_counter = 0
        for line in f_hypo:
            #meta = line.split("#") ## for joints included
            #bb_meta = meta[0].split(",")
            bb_meta = line.split(",")
            frame = int(bb_meta[0])
            bb_tlx = round(float(bb_meta[1]),1)
            bb_tly = round(float(bb_meta[2]),1)
            bb_w = round(float(bb_meta[3]),1)
            bb_h = round(float(bb_meta[4]),1)
                
            #joints = []
            #if(len(meta)>1):
            #    skeleton = meta[1].split(";")
            #    for j in skeleton:
            #        coords = j.split(",")   
            #        joints.append([round(float(coords[0]),1), round(float(coords[1]),1)])

            if(frame in HYPO):
                hps = HYPO[frame]
            else:
                hps = []

            #hps.append([bb_tlx,bb_tly,bb_w,bb_h,joints])
            hps.append([bb_tlx,bb_tly,bb_w,bb_h,])


            HYPO[frame] = hps

    return HYPO

# SAVE TRACKING OUTPUT        
def write_tracklets(file, tracklets, write_skeleton):
    with open(file, 'w') as output:
        for key in tracklets:
            id = key
            tracklet = tracklets[id]
            for detection in tracklet:
                frame = detection[0]
                joints = ""
                if(len(detection) == 6):
                    joints = detection[5]
                output.write("{},{},{},{},{},{}".format(frame, id, detection[1], detection[2], detection[3], detection[4]))
                if(write_skeleton):
                    output.write("#{}".format( joints))
                output.write("\n")

# SAVE TRACKING OUTPUT IN MOTCHALLENGE FORMAT
def write_tracklets_MOTChallenge(file, tracklets):
    with open(file, 'w') as output:
        for key in tracklets:
            id = key
            tracklet = tracklets[id]
            for detection in tracklet:
                frame = detection[0]
                joints = ""
                if(len(detection) == 6):
                    joints = detection[5]
                output.write("{}, {}, {}, {}, {}, {}, -1, -1, -1, -1 ".format(frame, id, round(detection[1],2), round(detection[2],2), round(detection[3],2), round(detection[4],2)))
                output.write("\n")


#5. DISTANCE FUNCTIONS
#=====================

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
    for i in range(0, len(joints1)):
        d += L2_dist(joints1[i],joints2[i])
    return d

def get_L2_dist_joints_bbox_weighted(joints1, joints2, weight):
    return get_L2_dist_joints(joints1, joints2) / weight

def IOU_multiple(group1, group2):
    costs = np.empty([len(group1), len(group2)], dtype=float)

    for i in range(0, len(group1)):
        for j in range(0, len(group2)):
            iou = bb_IoU(
                [group1[i][1], group1[i][2], group1[i][1] + group1[i][3], group1[i][2] + group1[i][4]],
                [group2[j][1], group2[j][2], group2[j][1] + group2[j][3], group2[j][2] + group2[j][4]]
            )
            costs[i][j] = iou

    rids, cids = solve_dense(costs)
    score =0
    matches = 0
    for r,c in zip(rids, cids):
        #print(r,c) # Row/column pairings
        score += costs[r][c]
        matches += 1

    return score/matches

# QUANTIFIES CONNECTIBILITY BETWEEN TRACKS t1 AND t2
# based on last match_basis frames, linearly projects track t1 by the number of frames between t1 end and t2 start (and also vice versa)
# projection creates match_frames frames, whose IOU with the target tracks beginning (end) is compared
def get_matching_score(t1, t2, match_basis, match_frames):
    score = 0

    s1, e1 = get_start_end_frame(t1)
    s2, e2 = get_start_end_frame(t2)

    # dynamic length matching
    l1 = e1-s1
    l2 = e2-s2
    match_basis_1 = min(l1, match_basis)
    match_basis_2 = min(l2, match_basis)
    #match_basis = MATCH_FRAMES
    #size_conf = 1-(1/match_basis)


    t1_tail_goal = t1[len(t1)-match_frames:]
    t2_head_goal = t2[:match_frames-1]

    t1_tail_basis = t1[len(t1)-match_basis_1:]
    t2_head_basis = t2[:match_basis_2-1]


    # TIME FORWARD PROJECTION
    frame_offset = s2 - e1
    projection1 = project(t1_tail_basis, frame_offset, match_frames)
    score1 = IOU_multiple(projection1, t2_head_goal)

    # TIME BACKWARD PROJECTION
    projection2 = project(t2_head_basis[::-1], frame_offset, match_frames)
    score2 = IOU_multiple(projection2, t1_tail_goal)

    return (score1 + score2)/2


#6. TRACKING
#===========
def track_munkres(H, iou_tracking, size_limit, cache_threshold):
    result = {} # {id:[frame, i],[frame+1, i], ...}
    tails = {} # {frame:{last_i:id}}
    orphans = {} # {frame: detection}
    ID = 0

    max_F = get_max_frame_number(H)
    for f in range(1,max_F-1):
        if((f-cache_threshold-1) in orphans):
            del(orphans[f-cache_threshold-1])

        if(not(f in H)):
            H[f] = []
        if(not(f+1 in H)):
            H[f+1] = []
        detections1 = H[f]
        detections2 = H[f+1]
        map_i_to_frame = []
        original_len = len(detections1)
        for i in range(0,original_len):
            map_i_to_frame.append(f)
        for frame in range(f-cache_threshold, f):
            if(frame in orphans):
                detections1.extend(orphans[frame])
                for i in range(0, len(orphans[frame])):
                    map_i_to_frame.append(frame)

        costs = np.empty([len(detections1), len(detections2)], dtype=float)
        for i in range(0, len(detections1)):
            for j in range(0, len(detections2)):
                iou = bb_IoU(
                    [detections1[i][0], detections1[i][1], detections1[i][0] + detections1[i][2], detections1[i][1] + detections1[i][3]],
                    [detections2[j][0], detections2[j][1], detections2[j][0] + detections2[j][2], detections2[j][1] + detections2[j][3]])
                costs[i][j] = -iou

        # MUNKRES
        matched_i = []
        rids, cids = solve_dense(costs)
        total = 0
        for i,j in zip(rids, cids):
            score = -costs[i][j]
            if(score < iou_tracking):
                continue

            frame_i = map_i_to_frame[i]
            if(frame_i < f):
                list = orphans[frame_i]
                for o in list:
                    if([o[0],o[1],o[2],o[3]] == [detections1[i][0],detections1[i][1],detections1[i][2],detections1[i][3]]):
                        i = o[4]
                        list.remove(o)
                        break
                orphans[frame_i] = list

            if(frame_i==f):
                matched_i.append(i)
            
            # case I: j links with existing track (i exist in tails)
            if(frame_i in tails):
                if(i in tails[frame_i]):
                    id = tails[frame_i][i]
                    result[id].append([f+1,j])
                    del(tails[frame_i][i])
                    if(not (f+1) in tails):
                        tails[f+1] = {}
                    tails[f+1][j] = id
                    continue


            # case II: i,j is a new pair (i does not exist in tails)
            id = ID
            ID += 1
            result[id] = [[frame_i,i],[f+1,j]]
            if(not (f+1) in tails):
                tails[f+1] = {}
            tails[f+1][j] = id



        for i in range(0, original_len):
            if(not(i in matched_i)):
                if(f in orphans):
                    list = orphans[f]
                else:
                    list = []
                d = detections1[i]
                d.append(i)
                list.append(d)
                orphans[f] = list


    tracks = {}
    for key in result:
        if(len(result[key])<size_limit):
            continue

        track = []
        fragment = []
        fragments = []
        frame_init = result[key][0][0]-1
        for t in result[key]:
            frame = t[0]
            if (not (frame == frame_init+1)):
                fragments.append(fragment)
                fragment = []
            temp = [frame]
            temp.extend(H[frame][t[1]])
            fragment.append(temp)
                    
            frame_init = frame
        fragments.append(fragment)
            
        if(len(fragments)==0):
            continue

        track = fragments[0]
        for i in range(1,len(fragments)):
            t1_start, t1_end = get_start_end_frame(track)
            t2_start, t2_end = get_start_end_frame(fragments[i])
            frame_gap = t2_start - t1_end
            last_frame = track[-1]
            first_frame = fragments[i][0]
            avg_width = (last_frame[3] + first_frame[3])/2
            avg_height = (last_frame[4] + first_frame[4])/2
            x_offset = (first_frame[1] - last_frame[1]) / frame_gap
            y_offset = (first_frame[2] - last_frame[2]) / frame_gap
            for frame in range (1, frame_gap):
                track.append([t1_end + frame, last_frame[1] + frame * x_offset, last_frame[2] + frame * y_offset, avg_width, avg_height]) 
            track.extend(fragments[i])

 
        tracks[key] = track 	#[[frame, b1, ..., b4], ... []]


    #print(tracks)
    return tracks


# 7. RE-ID
#============
def project(bboxes, frame_offset, length):
    projection = []

    avg_x = 0
    avg_y = 0
    avg_x_offset = 0
    avg_y_offset = 0
    avg_height = 0
    avg_width = 0
    #avg_size_ratio = 0

    for i in range(0, len(bboxes)-1):
        avg_width += bboxes[i][3]
        avg_height += bboxes[i][4]
        avg_x += bboxes[i][1]
        avg_y += bboxes[i][2]
        avg_x_offset += (bboxes[i+1][1] + (bboxes[i+1][3]/2)) - (bboxes[i][1] + (bboxes[i][3]/2))
        avg_y_offset += (bboxes[i+1][2] + (bboxes[i+1][4]/2)) - (bboxes[i][2] + (bboxes[i][4]/2))
        #avg_size_ratio += (bboxes[i+1][3]*bboxes[i+1][4])/(bboxes[i][3]*bboxes[i][4])
        if(i == len(bboxes)-2):
            avg_width += bboxes[i+1][3]
            avg_height += bboxes[i+1][4]
            avg_x += bboxes[i+1][1]
            avg_y += bboxes[i+1][2]

    avg_x = avg_x / len(bboxes)
    avg_y = avg_y / len(bboxes)
    avg_x_offset = avg_x_offset / (len(bboxes) - 1)
    avg_y_offset = avg_y_offset / (len(bboxes) - 1)
    avg_height = avg_height / len(bboxes)
    avg_width = avg_width / len(bboxes)
    #avg_size_ratio = avg_size_ratio / (len(bboxes)-1)
    #avg_size_ratio = (1+avg_size_ratio)/2 
    frame_offset = frame_offset + ((len(bboxes)-1)/2)

    anchor = [0, avg_x + (avg_x_offset * frame_offset), avg_y + (avg_y_offset * frame_offset), avg_width, avg_height]
    projection.append(anchor)
    for i in range (0, length-1): 
        projection.append([0, anchor[1] + (i+1)*avg_x_offset, anchor[2] + (i+1)*avg_y_offset, anchor[3], anchor[4]])
    
    return projection



def get_possible_pairings(tracklets, max_gap, match_basis, match_frames, min_length_to_match, match_score):
    pairings = {}

    for key1 in tracklets:
        for key2 in tracklets:
            if(key1 == key2):
                continue;

            t1 = tracklets[key1]
            t2 = tracklets[key2]

            t1_start, t1_end = get_start_end_frame(t1)
            t2_start, t2_end = get_start_end_frame(t2)

            l1 = t1_end - t1_start +1
            l2 = t2_end - t2_start +1
            min_length = min(l1, l2)
            frame_gap = 0

            if(t1_end < t2_start):
                frame_gap = t2_start - t1_end

            # PAIR WITH POTENTIAL MATCH
            if((frame_gap >= 1) and (frame_gap < max_gap) and (min_length > min_length_to_match)):
                d = get_matching_score(t1, t2, match_basis, match_frames)
                if(d>match_score):
                    id = str(key1) + "_" + str(key2)
                    pairings[id] = d

    return pairings


def match_pairings(pairings, tracklets):
    ignore_source = [] 
    ignore_target = []
    matches = {}

    for key, score in sorted(pairings.items(), key=lambda item: item[1], reverse=True):
        keys = key.split("_")
        source_id = int(keys[0]) 
        target_id = int(keys[1]) 
        if((source_id in ignore_source) or (target_id in ignore_target)):
             continue
        matches[source_id] = target_id
        ignore_source.append(source_id)
        ignore_target.append(target_id)



    streams = []
    for source in matches:
        stream = []
        stream.append(source)
        target = matches[source]
        stream.append(target)
        forward = True
        t = target
        while forward:
            if(t in matches):
                t = matches[t]
                stream.append(t)
            else:
                forward = False

        s = source
        backward = True
        while backward:
            if(s in matches.values()):
                for k, v in matches.items():
                    if v == s:
                        s = k
                        stream.insert(0, s)
            else:
                backward = False
                    
        if(not(stream in streams)):
            streams.append(stream)

    #print(streams)
    new_id = 0
    result = {}
    for stream in streams:
        for id in stream:
            if(new_id in result):
                sequence = result[new_id]

                # interpolate missing poses
                if(INTERPOLATE):  
                    t1_start, t1_end = get_start_end_frame(sequence)
                    t2_start, t2_end = get_start_end_frame(tracklets[id])
                    frame_gap = t2_start - t1_end
                    last_frame = sequence[-1]
                    first_frame = tracklets[id][0]
                    avg_width = (last_frame[3] + first_frame[3])/2
                    avg_height = (last_frame[4] + first_frame[4])/2
                    x_offset = (first_frame[1] - last_frame[1]) / frame_gap
                    y_offset = (first_frame[2] - last_frame[2]) / frame_gap
                    for frame in range (1, frame_gap):
                        sequence.append([t1_end + frame, last_frame[1] + frame * x_offset, last_frame[2] + frame * y_offset, avg_width, avg_height]) 

                sequence.extend(tracklets[id])
            else:
                sequence = tracklets[id]  
            
            result[new_id] = sequence


        new_id += 1


    for id in tracklets:
        if(id in ignore_source or id in ignore_target):
            continue
        result[new_id] = tracklets[id]
        new_id += 1

    return result


# 8. MISC / SUPOORTING METHODS
#=============================
def det_count(H):
    count = 0
    for key in H:
        count += len(H[key])
    return count                      

def center_diff(x1, x2):
    x_diff = x2[0] - x1[0]
    y_diff = x2[1] - x1[1]
    return math.sqrt((x_diff**2)+(y_diff**2))

def get_max_frame_number(dict):
    m = 0
    for a in dict:
        if (a > m):
            m = a
    return m

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


def stats(tracklets):
    t_count = 0
    bb_count = 0
    j_count = 0
    for id in tracklets:
        t_count += 1
        for bb in tracklets[id]:
            if(len(bb)>[5]):
                j_count += 1
            bb_count += 1
    return t_count, bb_count, j_count
    


def run_tracker(iou_tracking, size_limit, frame_gap, match_score, match_frames, match_basis, min_length_to_match,cache):
    if not os.path.exists(TRACK_OUTPUT_DIR):
        os.makedirs(TRACK_OUTPUT_DIR)

    for sequence in SEQUENCES_TRAIN:
        # SEQUENCE-LOCAL SETTINGS
        #iou_tracking = TRAIN_IOU_TRACKING[counter]
        #size_limit = TRAIN_SIZE_LIMIT[counter]
        #frame_gap = TRAIN_MATCH_MAX_FRAME[counter]
        #match_score  = TRAIN_MATCH_IOU[counter]
        #match_frames = TRAIN_MATCH_FRAMES[counter]
        #cache = TRAIN_CACHE_SIZE[counter]

        print(sequence, end = "\r")
        H_FILE = DET_DIR + sequence +".txt"
        H = get_hypotheses(H_FILE)
        tracklets = track_munkres(H, iou_tracking, size_limit, cache)
        pairings = get_possible_pairings(tracklets, frame_gap, match_basis, match_frames, min_length_to_match,match_score)
        tracklets = match_pairings(pairings, tracklets)


        write_tracklets(TRACK_OUTPUT_DIR + sequence + ".txt", tracklets, True)
        #write_tracklets_MOTChallenge(TRACK_OUTPUT_DIR + sequence + ".txt", tracklets)
        counter += 1


# MAIN:
def main():
    run_tracker(IOU_TRACKING, SIZE_LIMIT,  MATCH_MAX_FRAME_GAP, REQUIRED_MATCH_SCORE, MATCH_FRAMES, MATCH_BASIS, MIN_LENGTH_TO_MATCH)

if __name__ == "__main__":
    main()




######### END



# RUN TIME 
#def run_time():
#    run_tracker(IOU_TRACKING, SIZE_LIMIT,  MATCH_MAX_FRAME_GAP, REQUIRED_MATCH_SCORE, MATCH_FRAMES, MATCH_BASIS, MIN_LENGTH_TO_MATCH)
#print(timeit.timeit("run_time()", setup="from __main__ import run_time", number=3))


# GRID SEARCH
#param_1 = [0, 0.1, 0.2, 0.3]
#param_2 = [3,5,7]
#param_3 = [25, 50, 100]
#param_4 = [0.1, 0.2, 0.3, 0.4]
#param_5 = [2,3]

#for p1 in param_1:
#    for p2 in param_2:
#        for p3 in param_3:
#            for p4 in param_4:
#                for p5 in param_5:
#                    tracker = "sdp-ICPR/batch/sdp-iou{}-size{}-gap{}-score{}-frames{}".format(str(p1),str(p2),str(p3),str(p4),str(p5))
#                    TRACK_OUTPUT_DIR = WORK_SPACE_PATH + "data/mot17/tracklets/elias/" + tracker + "/"
#                    run_tracker(p1, p2,  p3, p4, p5, MATCH_BASIS, MIN_LENGTH_TO_MATCH)








