import motmetrics as mm
import numpy as np
from lapsolver import solve_dense

# working directory
WORK_SPACE_PATH = '/home/xelias3/deep-high-resolution-net.pytorch/'

# directory with ground truth (can be obtained from https://motchallenge.net/data/MOT17/)
GT_DIR= WORK_SPACE_PATH+'data/mot17/meta/ground-truth/'

# tracker dir name - directory name, where tracks are stored
tracker = "train/iou04-gap25-match06-cache10-munkres-reid/" 

# detection path
tracker_dir = WORK_SPACE_PATH+'data/mot17/tracks/' + tracker

# types of MOT17 objects that should be evaluated, the rest (bycicles cars, etc.) is avoided
ALLOWED_TYPES = [1,2,7,8,12]

# Only tracked detections that has overlap of at least IOU_LIMIT with ground truth can be evaluated as true positives
IOU_LIMIT = 0.5

# Test sequences of MOT17
SEQUENCES = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]

# Do not consider bounding boxes smaller than SIZE LIMIT, 0 - turn off
SIZE_LIMIT = 0
TOLERANCE = 0




def evaluateMOTA(seq_ids, size_limit, tolerance, tracker_dir, write_log):
    print(tracker_dir)
    accs = []
    nms = []

    counter = 0
    total = len(seq_ids)
    for seq_id in seq_ids:
        accs.append(get_MOTA_accumulator(seq_id, size_limit, tolerance, tracker_dir))
        nms.append(seq_id)
        counter += 1
        print("{}/{}".format(counter, total), end="\r")

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs, 
        metrics = mm.metrics.motchallenge_metrics, 
        names=nms,
        generate_overall=True)

    strsummary = mm.io.render_summary(
        summary, 
        formatters=mh.formatters, 
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)

    if(write_log):
        with open(WORK_SPACE_PATH+"data/mot17/logs/MOTA_log.txt", 'a') as output:
            output.write(tracker_dir)
            output.write(strsummary)



def get_MOTA_accumulator(seq_id, size_limit, tolerance, tracker_dir):

    data = get_data(seq_id, size_limit, tolerance, tracker_dir)
    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    for frame in data:
        acc.update(
            frame[0],   # Ground truth objects in this frame
            frame[1],   # Detector hypotheses in this frame
            frame[2]    # Distances from objects to hypotheses
        )

    return acc


def get_gt(gt_file, height_limit, height_tolerance, allowed_types):
    GT = {}
    NEUTRAL = {}

    with open(gt_file) as f_gt:
        poses_counter = 0
        for line in f_gt:

            meta = line.split(",")
            type = int(meta[7])
            frame = int(meta[0])
            id = int(meta[1])
            bb_tlx = int(meta[2])
            bb_tly = int(meta[3])
            bb_w = int(meta[4])
            bb_h = int(meta[5])
            bb_brx = bb_tlx+bb_w
            bb_bry = bb_tly+bb_h
            evaluate = int(meta[6])

            # SKIP GT WITH UNWANTED ANNOTATIONS SUCH AS CARS, BYCICLES etc.
            if(not(type in allowed_types)):
                continue

            # SKIP GT WITH TOO SMALL BOUNDING BOXES NOT ALLOWING SKELETON EXTRACTION
            if(bb_h < (height_limit-height_tolerance)):
                continue

            # FLAG BOXES THAT ARE ON THE SIZE BOUNDARY
            if((bb_h > (height_limit-height_tolerance)) and (bb_h < (height_limit+height_tolerance))):
                evaluate = 0

            # APPLICABLE GROUND TRUTHS GROUPED BY FRAME NUMBER
            if(evaluate == 0):
                if(frame in NEUTRAL):
                    neutrals = NEUTRAL[frame]
                else:
                    neutrals = []
                neutrals.append([bb_tlx,bb_tly,bb_w,bb_h])
                NEUTRAL[frame] = neutrals

            # NEUTRAL ANNOTATIONS THAT ARE NOT PENALIZED GROUPED BY FRAME NUMBER
            else:
                if(frame in GT):
                    gts = GT[frame]
                else:
                    gts = [] 
                gts.append([id,bb_tlx,bb_tly,bb_w,bb_h])
                GT[frame] = gts

    return [GT, NEUTRAL]

def get_hypotheses(hypo_file, size_limit):
    HYPO = {}
    
    with open(hypo_file) as f_hypo:
        poses_counter = 0
        for line in f_hypo:
            bbox = line.split("#")[0]
            meta = bbox.split(",")
            frame = int(meta[0])
            id = meta[1]
            bb_tlx = float(meta[2])
            bb_tly = float(meta[3])
            bb_w = float(meta[4])
            bb_h = float(meta[5])

            if(bb_h < size_limit):
                continue

            if(frame in HYPO):
                hps = HYPO[frame]
            else:
                hps = []


            hps.append([id,bb_tlx,bb_tly,bb_w,bb_h])
            HYPO[frame] = hps

    return HYPO

def get_max_frame_number(dict):
    m = 0
    for a in dict:
        if (a > m):
            m = a
    return m


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
        iou = np.nan

    if(iou<0.5):
        iou = np.nan

    return iou

def check_eval(i, neutral):
    e = True
    for n in neutral:
        iou = bb_IoU([i[1],i[2],i[1]+i[3],i[2]+i[4]],[n[0],n[1],n[0]+n[2],n[1]+n[3]])
        if(iou > IOU_LIMIT):
            e = False
    return e


def get_data(seq_id, size_limit, tolerance, tracker_dir):         
    h = get_hypotheses(tracker_dir+seq_id+".txt", size_limit)
    gt = get_gt(GT_DIR+seq_id+".txt", size_limit, tolerance, ALLOWED_TYPES)
    gt_qualified = gt[0]
    gt_neutral = gt[1]
    max_frame = max(get_max_frame_number(gt_qualified),get_max_frame_number(h))

    data = []
    for frame in range (1, max_frame+1):
        GTs = []
        Hs = []
        Ds = []
    
        m = []
        n = []

        if(frame in gt_qualified):
            for i in gt_qualified[frame]:
                GTs.append(i[0])
                m.append(i)

        if(frame in h):
            for i in h[frame]:
                neutral = []
                if(frame in gt_neutral):
                    neutral = gt_neutral[frame]
                if(check_eval(i,neutral)):
                    Hs.append(i[0])
                    n.append(i)

        Ds = np.zeros((len(m), len(n)))
        for i in range (0,len(m)):
            for j in range (0, len(n)):
                Ds[i][j] = bb_IoU([m[i][1],m[i][2],m[i][1]+m[i][3],m[i][2]+m[i][4]],[n[j][1],n[j][2],n[j][1]+n[j][3],n[j][2]+n[j][4]])

        data.append([GTs,Hs,Ds])

    return data

evaluateMOTA(SEQUENCES, SIZE_LIMIT, TOLERANCE, tracker_dir, False)




# BATCH EVALUATION
#param_1 = [0.3, 0.4, 0.5]
#param_2 = [3,5,7]
#param_3 = [25, 50, 100]
#param_4 = [0.5, 0.6]
#param_5 = [3,5,10]
#total = len(param_1)*len(param_2)*len(param_3)*len(param_4)*len(param_5)
#counter = 0
#for p1 in param_1:
#    for p2 in param_2:
#        for p3 in param_3:
#            for p4 in param_4:
#                for p5 in param_5:
#                    tracker = "sdp-MTAP/batch/sdp-iou{}-size{}-gap{}-score{}-cache{}".format(str(p1),str(p2),str(p3),str(p4),str(p5))
#                    tracker_dir = WORK_SPACE_PATH + "data/mot17/tracks/" + tracker + "/"
#                    evaluateMOTA(SEQUENCES, SIZE_LIMIT, TOLERANCE, tracker_dir, True)
#                    print(str(100*counter/total) + " % done", end = "\r")
