# Tracking subjects and detecting relationships in crowded videos 
*Petr ELIAS. Matus MACKO, Jan SEDMIDUBSKY, Pavel ZEZULA*  
*MTAP-D-20-01870 (under review)*

Multi-subject tracking in crowded videos is a challenging research direction with high applicability in robotic vision, public safety, crowd management, autonomous driving vehicles, or psychology. This work proposes a near-online real-time tracking approach based on bounding-box detection association in consecutive video frames. Our work is inspired in popular methods [][], however, a significant reduction in track fragmentation and identity switching is achieved by the proposed re-identification phase. We further demonstrate the tracker applicability in human relationships detection scenario without utilizing visual features.

## Pipeline
![Flowchart](/supplementary/flowchart.png "Tracking ad detection flowchart")
**1. Object detection** *(in each image)*
For MOT17/MOT20 the detection are already provided. Own detection methods can be used (e.g., yolo).
**2. Pose estimation** *(in each image)*
We used hrnet for pose estimation on the detected areas obtained in Step 1. Other methods can be used
**3. Tracking** *(in each image sequence)* 
We propose greedy (sub-optimal) and Munkres (optimal) association algorithms. Both versions are enhanced with unassociated detection caching. Re-identification method based on track mutual projection can be optionally turned on to reduce track fragmentation. See details below.
**4. Entitative relationship detection** *(in set of tracks)*
Computation of hand distance and body proportion features is used to detect pairs holding hands and children in the video

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
