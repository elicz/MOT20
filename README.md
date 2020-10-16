# Tracking subjects and detecting relationships in crowded videos 
*Petr ELIAS. Matus MACKO, Jan SEDMIDUBSKY, Pavel ZEZULA*  
*MTAP-D-20-01870 (under review)*

Multi-subject tracking in crowded videos is a challenging research direction with high applicability in robotic vision, public safety, crowd management, autonomous driving vehicles, or psychology. This work proposes a near-online real-time tracking approach based on bounding-box detection association in consecutive video frames. Our work is inspired in popular methods \[1,2\], however, a significant reduction in track fragmentation and identity switching is achieved by the proposed re-identification phase. We further demonstrate the tracker applicability in human relationships detection scenario without utilizing visual features.

## Pipeline
![Flowchart](/supplementary/flowchart.png "Tracking ad detection flowchart")
**1. Video containing subjects**  
We used the videos from [MOT17](https://motchallenge.net/data/MOT17)\[3\] and [MOT20](https://motchallenge.net/data/MOT20)\[4\] datasets, but any video(s) containing people (even crowded video) would be good.  
  
**2. Object detection** *(in each image)*  
For MOT17/MOT20 the detection are already provided. Own detection methods can be used (e.g., YOLO\[5\]).  
  
**3. Tracking** *(in each image sequence)*   
We propose greedy (sub-optimal) and Munkres (optimal) association algorithms. Both versions are enhanced with unassociated detection caching. Re-identification method based on track mutual projection can be optionally turned on to reduce track fragmentation. [See details below](#tracking)
  
**4. Pose estimation** *(in each image)*  
We used hrnet for pose estimation on the detected areas obtained in Step 1. Other methods can be used.  
  
**5. Entitative relationship detection** *(in set of tracks)*  
Computation of hand distance and body proportion features is used to detect pairs holding hands and children in the video. [See details below](#rel_det).

## <a name="tracking"></a>Tracking
Two generic tracking algorithms are used for bounding box association in consecutive frames: i) optimal Munkres and 2) Sub-optimal Greedy variant. On MOT17 they have similar performance in terms of accuracy and efficiency. Both methods are enhanced with unassociated detection caching for (default) duration of `cache=7` frames. Set `cache=0` for NO caching, or adjust accordingly. All [tracking parameters](track_params) are specified below in more detail.
> Tracking methods are implemented in *multi-object-tracker.py*

### Dependencies
Following Python libraries are used:
```
    numpy
    lapsolver
    heapq
```

### Input format
The `get_hypotheses("file_with_detections.txt")` takes detection file in the following format (i.e., each detection \[frame_nr,x,y,w,h\] on a seperate line):
```
    d1_bbox_frame_number,d1_bbox_top_left_corner_x,d1_bbox_top_left_corner_y,d1_bbox_width,d1_bbox_height
    d2_bbox_frame_number,d2_bbox_top_left_corner_x,d2_bbox_top_left_corner_y,d2_bbox_width,d2_bbox_height
    ...
```
*Detections text files in MOT17 use sightly different format (there is additional attribute thath needs to be skipped). Method `get_hypotheses(` need to be adjusted accordingly.*

### Running the tracker
To run the tracker on the detection file, use:
```
    H = get_hypotheses(file_with_detections)  ## load detections
    tracks = track_munkres(H, iou_tracking, size_limit, cache)  ## optimal tracking variant (slightly slower)
```
or
```
    tracks = track_greedy(H, iou_tracking, size_limit, cache) ## sub-otpimal greedy variant (slightly faster, similar results)
```

#### Re-Identification

#### <a name="track_params"></a>Tracking parameters
Also, you can adjust the tracking parameters:
```
# 3. TRACKING PARAMETERS
#========================
IOU_TRACKING = 0.2        # IOU tracking limit to match two detections, default=0.25
SIZE_LIMIT = 3            # Minimum number of frames required to constitute a track, default=5
INTERPOLATE = True        # Interpolate poses in re-identified tracks, default=True
MIN_LENGTH_TO_MATCH = 3   # Minimum length of track required for matching fragmented tracks, default=3
MATCH_FRAMES = 2          # Exact number of frames to be projected for matching fragmented tracks, default=2 (event. 3) 
MATCH_BASIS = 30      #If fragmented track has more frames than MIN_LENGTH_TO_MATCH, maximum number of frames to take into account when projecting (minimum of (length,match_basis is taken), default=30
MATCH_MAX_FRAME_GAP = 50  # Max allowed gap between to-be-matched fragmented tracklets, default=50 
REQUIRED_MATCH_SCORE = 0.3# Min IOU score to match two fragmented tracks based on MATCH_FRAMES X MATCH_FRAMES sum of IOU, default=0.25 
```

### Results

## <a name="rel_det"></a> Entitative Relationship Detection
> Tracking methods are implemented in *group_detecion_and_search.py*

### Dependencies
Following Python libraries are used:
```
    numpy
    cv2
    statistics
```

### Input format
### Running the detection
### Results

## Reference
For referencing this work please use the following citation:

## Bibliohraphy
\[1\] Bewley, A., Ge, Z., Ott, L., Ramos, F.T., Upcroft, B.: Simple online and realtime tracking. In: 2016 IEEE In-ternational Conference on Image Processing, ICIP 2016, Phoenix, AZ, USA, September  25-28, 2016, pp. 3464–3468. IEEE (2016)  
\[2\] Bochinski, E., Eiselein,  V., Sikora, T.: High-speed tracking-by-detection without using image information. In: 14th IEEE International Conference on Advanced Video and Signal Based Surveillance, AVSS 2017, Lecce, Italy, August 29 - September 1, 2017, pp. 1–6. IEEE Com-puter Society (2017)  
\[3\] Milan,  A.,  Leal-Taixe,  L.,  Reid,  I.D.,  Roth, S.,Schindler, K.: MOT16: A benchmark for multi-object tracking. CoRRabs/1603.00831(2016). URL http://arxiv.org/abs/1603.00831  
\[4\] Dendorfer, P., Rezatofighi, H., Milan, A., Shi, J., Cremers, D., Reid, I., Roth, S., Schindler, K., Leal-Taixe, L.: Mot20: A benchmark for multi object tracking in crowded  scenes. arXiv:2003.09003[cs] (2020). URL http://arxiv.org/abs/1906.04567. ArXiv: 2003.09003  
