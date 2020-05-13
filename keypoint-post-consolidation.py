'''
Consolidation the bee's head keypoint labeling result of three workers
The labling job is posted on AWS Sagemaker Grounf Truth
reference: Dr Ann Kennedy's code and instruction https://github.com/annkennedy/asm-workflow
'''

import json

import numpy as np
from pathlib import Path
import os
from collections import defaultdict
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment


def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return (int(sum_x // length), int(sum_y // length))


def confidence_function(d, scale=0.0035):
    return np.clip(2 - np.exp(scale * d), 0, 1)

'''

# store all data into same json file

# set the path to the downloaded data:
data_path = './S3_bee_happy_bucket/Annotations/test-keypoint-100demo/annotations/consolidated-annotation/consolidation-response/iteration-1'
#
# # set a path for saving predictions:
# preds_path = 'data/hw01_preds'
# os.makedirs(preds_path,exist_ok=True) # create directory if needed

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.json' in f]
data = []

for file_name in file_names:
    for line in open(Path(data_path) / file_name, 'r'):
        line_output = json.loads(line)
        for d in line_output:
            data.append(d)

# store res, overwrte the previous result
with open('test100-all-annotations.json', 'w') as outfile:
    json.dump(data, outfile)
'''

# first read the json
demo_path = './S3_bee_happy_bucket/Annotations/test-keypoint-100demo/test100-all-annotations.json'
data = []
for line in open(demo_path, 'r'):
    data = json.loads(line)

n_images = len(data)

# load json data to annotations dict
annotations = defaultdict(lambda: defaultdict(lambda: []))  # annotations[image][worker] = [(x1,y1),...(xn,yn)]

for i, image in enumerate(data):
    id = int(image['datasetObjectId'])
    workers = image["consolidatedAnnotation"]["content"]["test-keypoint-100demo"][
        'annotationsFromAllWorkers']  # list of dictionary [{}], each dict is one work
    for j, worker in enumerate(workers):
        annot = eval(worker['annotationData']['content'])
        keypoints = annot['annotatedResult']['keypoints']  # list of dictionary [{}], each dict is one point
        for keypoint in keypoints:
            x = keypoint['x']
            y = keypoint['y']
            annotations[id][j].append((x, y))

# associate, dealing with annotations dict
weight_4numOfValidPoint = [0.1, 0.7, 1]
consolidated_com = defaultdict(lambda: defaultdict(lambda: []))
consolidated_res = defaultdict(lambda: [])

for i in range(n_images):
    # pick the reference annotations
    reference_annotator = np.argmax(np.asarray([len(annotations[i][j]) for j in range(3)]))
    reference_annotations = annotations[i][reference_annotator]
    consolidated_com[i] = {a: [] for a in reference_annotations}

    other_worker_list = [0, 1, 2]
    other_worker_list.remove(reference_annotator)

    for worker in other_worker_list:
        worker_annotations = annotations[i][worker]  # list of (x,y)
        if not worker_annotations:
            continue

        distances = distance.cdist(reference_annotations, worker_annotations)
        reference_idxs, matched_idxs = linear_sum_assignment(distances)

        for n in range(len(reference_idxs)):
            reference_annotation = reference_annotations[reference_idxs[n]]
            matched_annotation = worker_annotations[matched_idxs[n]]
            consolidated_com[i][reference_annotation].append(matched_annotation)

    # consolidate
    # others: list with max length 2 storing the associated points' idx
    for ref, others in consolidated_com[i].items():
        if not others:
            d = 0
            coe_idx = 0
        else:
            distances = distance.cdist([ref], others)
            d = np.mean(distances)
            coe_idx = len(distances)

        confidence_value = weight_4numOfValidPoint[coe_idx] * confidence_function(d)
        others.append(ref)
        consolidated_res[i].append((centeroidnp(np.asarray(others)), float(confidence_value)))

# store res, overwrte the previous result
with open('/Users/Evelyn/Desktop/EE148CV/bee-happy/S3_bee_happy_bucket/Annotations/test-keypoint-100demo/consolidated_result.json', 'w') as outfile:
    json.dump(consolidated_res, outfile)
