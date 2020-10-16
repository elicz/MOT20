# Tracking Subjects and Detecting Relationships in Crowded City Videos 
Petr ELIAS. Matus MACKO, Jan SEDMIDUBSKY, Pavel ZEZULA
MTAP-D-20-01870 (under review)

Multi-subject tracking in crowded videos is an established yet challenging research direction with high applicability in public safety, crowd management, urban planning, autonomous driving vehicles, robotic vision, and psychology. This work proposes a near-online real-time tracking approach based on bounding-box detection association in consecutive video frames. Our work is inspired in popular methods [][], however, we achieve dramatic reduction in track fragmentation and identity switching by the re-identification phase. We also demonstrate the tracker applicability in human relationships detection scenario -- based on the extracted joint position information we detect couples holding hands and children in crowded videos, without utilizing any visual features.

## Pipeline
img
1. Object detection (in each image) - for MOT17/MOT20 the detection are already provided. Own detection methods can be used (e.g., yolo).
2. Pose estimation (in each image) - we used hrnet for pose estimation on the detected areas obtained in Step 1. Other methods can be used
3. Tracking (in each image sequence) - we propose greedy (sub-optimal) and Munkres (optimal) association algorithms. Both versions are enhanced with unassociated detection caching. Re-identification method based on track mutual projection can be optionally turned on to reduce track fragmentation. See details below.
4. Entitative relationship detection (in set of tracks) - computation of hand distance and body proportion features is used to detect pairs holding hands and children in the video

## Tracking
Tracking methods are implemented in 
### Dependencies
### Input format
### Running the tracker
### Results

## Entitative Relationship Detection
Detection and group discvoery methods can be found in
### Dependencies
### Input format
### Running the detection
### Results

## Reference
For referencign this work please use the following citation:
