'''
Consolidation the bee's head keypoint labeling result of three workers
The labling job is posted on AWS Sagemaker Grounf Truth
reference: Dr Ann Kennedy's code and instruction https://github.com/annkennedy/asm-workflow
'''

import json
import numpy as np
from scipy.spatial.distance import cdist, euclidean
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from collections import defaultdict

from pathlib import Path
from scipy.spatial import distance


def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

path = Path('./S3_bee_happy_bucket/Annotations')

demo_path = path/'demo/.manifest'

# first read the json
data = []
for line in open(demo_path,'r'):
    data.append(json.loads(line))

n_images = len(data)

# load json data
annotations = defaultdict(lambda:defaultdict(lambda :[])) #annotations[image][worker] = [(x1,y1),...(xn,yn)]
visited = defaultdict(lambda:defaultdict(lambda :[]))

for i,image in enumerate(data):
    workers = image['annotatedResult']['annotationsFromAllWorkers'] # list of dictionary [{}], each dict is one work
    for j,worker in enumerate(workers):
        keypoints = worker['annotationData']['content']['annotatedResult']['keypoints'] # list of dictionary [{}], each dict is one point
        x = keypoints['x']
        y = keypoints['y']
        annotations[i][j].append((x,y))
        visited[i][j].append(False)

# associate
threshold = 10
weight_4numOfValidPoint = [0.1,0.7,1]
consolidated_res = defaultdict(lambda:[])

for i in range(n_images):
    for j in range(3):
        for k, keypoint in enumerate(annotations[i][j]):
            keypoint = np.asarray(keypoint)
            if visited[i][j][k]:
                continue
            visited[i][j][k] = True

            other_worker_list = [0,1,2]
            other_worker_list = other_worker_list.remove(j)

            # check other workers' result
            candidate_points = [keypoint]
            associate_distances = []
            for worker in other_worker_list:
                worker_annotations = np.asarray(annotations[i][worker]) # list of (x,y)
                distance = distance.cdist(keypoint,worker_annotations)
                min_distance = np.min(distance)
                min_idx = np.argmin(distance)

                if visited[i][worker][min_idx] or min_distance > threshold:
                    continue

                visited[i][worker][min_idx] = True
                candidate_points.append(annotations[i][worker][min_idx])
                associate_distances.append(min_distance)

            associated_point = centeroidnp(np.asarray(candidate_points))

            confidence_value = weight_4numOfValidPoint[len(associate_distances)]*(1/(1+np.mean(associate_distances)))

            consolidated_res[i].append((associated_point,confidence_value))

# store res, overwrte the previous result
with open('consolidated_res.txt', 'w') as outfile:
    json.dump(consolidated_res, outfile)
