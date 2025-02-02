import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import glob
import time
import cv2

import requests
import json
from datetime import datetime

#imports for dlib
from imutils.video import FPS
import numpy as np
import argparse
import imutils
#import dlib
import cv2
import random

# Hungarian assignment
#from sklearn.utils.linear_assignment_ import linear_assignment
#from scipy.optimize.linear_sum_assignment import linear_sum_assignment
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

#ROOT_DIR = os.path.abspath("../")
# Import Mask RCNN
#sys.path.append(ROOT_DIR)  # To find local version of the library
#from mrcnn import utils
#import mrcnn.model as modellib
#from mrcnn import visualize
# Import COCO config
#sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
#import coco

# Directory to save logs and trained model
#MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

#class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
#    GPU_COUNT = 1
#    IMAGES_PER_GPU = 1

#config = InferenceConfig()
#config.display()

# Create model object in inference mode.
#model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load weights trained on MS-COCO
#model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("-d", "--det_file", type=str,
	help="path to detection txt file")
ap.add_argument("-mask", "--mask_path", type=str,
	help="path to mask txt file")

args = vars(ap.parse_args())

mask = cv2.imread(args["mask_path"])
h = mask.shape[0]
w = mask.shape[1]

#for y in range(0, h):
#    for x in range(0, w):
#        if mask[y,x,0] == 0 and mask[y,x,1] == 0 and mask[y,x,2] == 0:
#            continue
#        else:
#            mask[y,x,0] = 255
#            mask[y,x,1] = 255
#            mask[y,x,2] = 255

cv2.imwrite('_mask.jpg', mask)

# initialize the video stream and output video writer
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["video"])
writer = None

# initialize the list of object trackers and corresponding class
# labels
trackers = []
labels = []

# start the frames per second throughput estimator
fps = FPS().start()
frame_number = 0

EUCL_THRESH = 35
DIST_INFINITE = 100000

# import the necessary packages
from collections import namedtuple
import numpy as np
import cv2

# define the `Detection` object
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])

def print_id_by_index(tracker_lst, index_lst):
    id_lst = []
    for index in index_lst:
        t, l = tracker_lst[index]
        id_lst.append(l)
    print(id_lst)

def detection_step(frame_number):
    return frame_number == 1 or frame_number % 6 == 0

def draw_track(frame, d, l, color=(255,0,0)):
    cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), color, 2)
    cv2.putText(frame, l, (d[0], d[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

def is_track_pos_overlap(frame, tracker_lst, det_bbox):
    # check if an existing track is around current detection
    det_tup = (det_bbox[0], det_bbox[1], det_bbox[2], det_bbox[3])

    for trobj in tracker_lst:
        t, l = trobj
        _, pos = t.update(frame)
        (x,y,w,h) = pos
        tr_tup = (x,y,x+w,y+h)

        iou = bb_intersection_over_union(tr_tup, det_tup)

        if iou > 0.1:
            return True    
    return False

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

def _assign_new_track(d, tracker_lst):
    new_label = random.sample(range(50, 100), 1)
    new_track = cv2.TrackerCSRT_create()
    new_track.init(frame, (d[0], d[1], d[2]-d[0], d[3]-d[1]))
    tracker_lst.append((new_track, str(new_label[0])))

def _group(detections):
    candidates = []
    for i in range(len(detections)):
        if i in candidates:
            continue

        for j in range(len(detections)):
            if i != j and j not in candidates:
                if bb_intersection_over_union(detections[i], detections[j]) > 0.0:
                    candidates.append(i)
                    candidates.append(j)
    return candidates

def all_same(items):
    return all(x == items[0] for x in items)

def _distance(pt1, pt2):
    dist = np.linalg.norm(pt1-pt2)
    return dist

def euclidean_dist(frame, track, detections):
    t, l = track
    _dist_cur_track = np.zeros(len(detections))
    
    _, bbox = t.update(frame)
    (x, y, w, h) = bbox
    
    bx = x + w / 2
    by = y + h

    for i in range(len(detections)):
        det_pos = detections[i]
    
        d_w = det_pos[2] - det_pos[0]

        dbx = det_pos[0] + d_w / 2 
        dby = det_pos[3]

        pt1 = np.asarray([bx, by], dtype=np.float)
        pt2 = np.asarray([dbx, dby], dtype=np.float)
        _dist_cur_track[i] = _distance(pt1, pt2)
    
    return _dist_cur_track

def compute_cost_matrix(frame, trackers, detections):
    # compute coordinates of bottom middle point for both trackers and detections
    # create list of tuples of coordinates
    track_coords = []
    det_coords = []

    cost_mtx = np.zeros((len(trackers), len(detections)))

    for tobj in trackers:
        t, lbl = tobj
        _, pos = t.update(frame)
        (x,y,w,h) = pos
        track_coords.append((x+w/2,y+h))
    
    for det in detections:
        w = (det[2]-det[0])
        det_coords.append((det[0]+w/2,det[3]))
        
    cost_mtx = distance.cdist(track_coords, det_coords, 'euclidean')
    cost_mtx[cost_mtx > EUCL_THRESH] = DIST_INFINITE
    return cost_mtx


multi_tracker = cv2.MultiTracker_create()
tracker_lst = []
#unmatched_tracks = set()
unmatched_dets = set()
redundant_tracks = set()

overlaping_tracks = []

while True:
    frame_number = frame_number + 1
    # grab the next frame from the video file
    print('[DEBUG] --> reading frame ' + str(frame_number))
    (grabbed, frame) = vs.read()
    
    # check to see if we have reached the end of the video file
    if frame is None:
        break

    #frame = cv2.bitwise_and(frame, mask)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    out_size = (1980, 480)
    # if we are supposed to be writing a video to disk, initialize
    # the writer
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
            out_size, True)

    # if there are no object trackers we first need to detect objects
    # and then create a tracker for each object
    id_labels = random.sample(range(1, 1000), 100)

    if detection_step(frame_number) == True:
        detections = []

        # reading detections from txt file
        with open(args["det_file"], "r") as f:
            for line in f:
                currentline = line.split(",")
                frame_num = int(currentline[0])

                x1 = int(float(currentline[1]))
                y1 = int(float(currentline[2]))
                x2 = int(float(currentline[3])) 
                y2 = int(float(currentline[4]))

                if frame_num == frame_number:
                    #cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                    detections.append([x1,y1,x2,y2])

        # removing overlaping detections
        group_candidate = _group(detections)
        detections = [detections[i] for i in range(len(detections)) if i not in group_candidate]

        if frame_number == 1:
            for i in range(len(detections)):    
                box = detections[i]
                label = str(i)

                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (box[0], box[1], box[2]-box[0], box[3]-box[1]))
                tracker_lst.append((tracker, label))

                draw_track(frame, box, label, (0,0,255))
        else:
            active_tracks_index = []
            remove_rows = []

            for box in detections:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            cost_mtx = compute_cost_matrix(frame, tracker_lst, detections)
            
            for row in range(cost_mtx.shape[0]):
                if np.all(cost_mtx[row] == DIST_INFINITE, axis=0):
                    remove_rows.append(row)
                else:
                    active_tracks_index.append(row)

            cost_mtx = np.delete(cost_mtx, remove_rows, axis=0)
            row_ind, col_ind = linear_sum_assignment(cost_mtx)

            for i in range(len(detections)):
                if i not in col_ind:
                    unmatched_dets.add(i)

            for row, col in zip(row_ind, col_ind):
                t, label = tracker_lst[active_tracks_index[row]]
                d = detections[col]

                dist = euclidean_dist(frame, tracker_lst[active_tracks_index[row]], [d])
                if dist[0] > EUCL_THRESH:   
                    unmatched_dets.add(col)
                    continue

                new_track = cv2.TrackerCSRT_create()
                new_track.init(frame, (d[0], d[1], d[2]-d[0], d[3]-d[1]))
                tracker_lst[active_tracks_index[row]] = (new_track, label)

                draw_track(frame, d, label, (0,0,255))

                if active_tracks_index[row] in redundant_tracks:
                    redundant_tracks.remove(active_tracks_index[row])
            
            track_match_temp = []
            # assign redundant tracks to closest of unmatched detections
            for index in redundant_tracks: 
                _, label = tracker_lst[index]

                dists = []
                dist2index = {}
                
                for det_index in unmatched_dets:
                    d = detections[det_index]
                    dist = euclidean_dist(frame, tracker_lst[index], [d])
                    
                    dists.append(dist[0])
                    dist2index[dist[0]] = det_index

                    if len(dists) == 0: continue
                    # find min in dists & assign current trobj to it
                    mindist = min(dists)

                    if mindist > 100: continue

                    min_det_index = dist2index[mindist]
                    det = detections[min_det_index]

                    new_track = cv2.TrackerCSRT_create()
                    new_track.init(frame, (det[0], det[1], det[2]-det[0], det[3]-det[1]))
                    tracker_lst[index] = (new_track, label)

                    track_match_temp.append(index)

            for index in track_match_temp:
                redundant_tracks.remove(index)

            for det_index in unmatched_dets:
                if not is_track_pos_overlap(frame, tracker_lst, detections[det_index]):
                    _assign_new_track(detections[det_index], tracker_lst)
            
            unmatched_dets.clear()
    else:
        tlbr_pos_lst = []
        pos2index = {}

        print_id_by_index(tracker_lst, redundant_tracks)

        for index in range(len(tracker_lst)):
            if index in redundant_tracks:   continue

            track, l = tracker_lst[index]
            _, bbox = track.update(frame)

            tlbr = (int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
            tlbr_pos_lst.append(tlbr)
            pos2index[tlbr] = index

            draw_track(frame, tlbr, l)

        for i in range(len(tlbr_pos_lst)):
            pos1 = tlbr_pos_lst[i]
            for j in range(i+1, len(tlbr_pos_lst)):                
                pos2 = tlbr_pos_lst[j]
                iou = bb_intersection_over_union(pos1, pos2)
                if iou > 0.55:
                    redundant_tracks.add(pos2index[pos2])
        
    frame = cv2.resize(frame, out_size)
    cv2.putText(frame, 'frame :'+str(frame_number), (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

    if writer is not None:
        print('[DEBUG] --> writing frame')
        writer.write(frame)

    # show the output frame
    #cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()