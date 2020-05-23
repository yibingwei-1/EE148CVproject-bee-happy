'''
Consolidation the bee's head keypoint labeling result of three workers
The labling job is posted on AWS Sagemaker Grounf Truth
reference: Dr Ann Kennedy's code and instruction https://github.com/annkennedy/asm-workflow
'''

import json

import numpy as np
import os
from collections import defaultdict
from scipy.spatial import distance
from pathlib import Path

def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return (int(sum_x // length), int(sum_y // length))


def confidence_function(d, scale=0.0035):
    return np.clip(2 - np.exp(scale * d), 0, 1)

def get_mean_annotations_from_dict(consolidated_com):
    res = []
    for ref, others in consolidated_com.items():
        if not others:
            d = 0
            coe_idx = 0
        else:
            distances = distance.cdist([ref], others)
            d = np.mean(distances)
            coe_idx = len(distances)

        #confidence_value = weight_4numOfValidPoint[coe_idx] * confidence_function(d)
        others.append(ref)
        res.append((centeroidnp(np.asarray(others)))) #, float(confidence_value)))

    return res


# store all data into same json file

# set the path to the downloaded data:
dir_path = Path('../bee-happy-bucket/Annotations/Step1/Bee-Happy-True-Step1/')
#
# # set a path for saving predictions:
# preds_path = 'data/hw01_preds'
# os.makedirs(preds_path,exist_ok=True) # create directory if needed

# get sorted list of files:
data_path = dir_path/'annotations/consolidated-annotation/consolidation-response/iteration-1'
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
with open(dir_path/'annotations/all-annotations.json', 'w') as outfile:
    json.dump(data, outfile)


# first read the json
demo_path = dir_path/'annotations/all-annotations.json'
data = []
for line in open(demo_path, 'r'):
    data = json.loads(line)

n_images = len(data)

# load json data to annotations dict
annotations = defaultdict(lambda: defaultdict(lambda: []))  # annotations[image][worker] = [(x1,y1),...(xn,yn)]
visited = defaultdict(lambda:defaultdict(lambda :[])) #

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
            visited[id][j].append(False)

# associate, dealing with annotations dict
weight_4numOfValidPoint = [0.1, 0.7, 1]
consolidated_res = defaultdict(lambda: [])

threshold = 45


for i in range(n_images):
    order = list(np.argsort(np.asarray([len(annotations[i][w]) for w in range(3)])))
    for j in order[::-1]:
        for k, keypoint in enumerate(annotations[i][j]):
            #keypoint = np.asarray(keypoint)
            if visited[i][j][k]:
                continue
            visited[i][j][k] = True

            other_worker_list = [0,1,2]
            other_worker_list.remove(j)

            # check other workers' result
            candidate_points = [keypoint]
            associate_distances = []

            for worker in other_worker_list:
                worker_annotations = annotations[i][worker] # list of (x,y)

                if len(worker_annotations) < 1:
                    continue

                distances = distance.cdist([keypoint],worker_annotations)
                min_distance = np.min(distances)
                min_idx = np.argmin(distances)

                if visited[i][worker][min_idx] or min_distance > threshold:
                    continue

                visited[i][worker][min_idx] = True
                candidate_points.append(annotations[i][worker][min_idx])
                associate_distances.append(min_distance)

            associated_point = centeroidnp(np.asarray(candidate_points))

            confidence_value = weight_4numOfValidPoint[len(associate_distances)]*(1/(1+np.mean(associate_distances)))

            consolidated_res[i].append((associated_point,confidence_value))

# store res, overwrte the previous result
with open(dir_path/'annotations/consolidated-annotations.json', 'w') as outfile:
    json.dump(consolidated_res, outfile)
'''
for i in range(n_images):

    # pick the reference annotations
    annotator3, annotator2, annotator1  = np.argsort(np.asarray([len(annotations[i][j]) for j in range(3)]))
    # annotator1 is the reference annotator with the most annotations and annotator3 annotate the least

    annotations1 = annotations[i][annotator1]
    annotations2 = annotations[i][annotator2]
    annotations3 = annotations[i][annotator3]

    consolidated_com_23 = {a: [] for a in annotations2}
    consolidated_com_all = {a: [] for a in annotations1}

    if not annotations1:
        continue

    if annotations2 and annotations3:

        distances = distance.cdist(annotations2, annotations3)
        worker2_idxs, worker3_idxs = linear_sum_assignment(distances)

        for n in range(len(worker2_idxs)):
            keypoint2 = annotations2[worker2_idxs[n]]
            keypoint3 = annotations3[worker3_idxs[n]]

            consolidated_com_23[keypoint2].append(keypoint3)


        annotations23 = get_mean_annotations_from_dict(consolidated_com_23)
        # add the annotations that are captured by 3 but are captured by2
    else:
        annotations23 = annotations2

    if annotations23:
        distances = distance.cdist(annotations1, annotations23)
        worker1_idxs, worker23_idxs = linear_sum_assignment(distances)

        for n in range(len(worker1_idxs)):
            keypoint1 = annotations1[worker1_idxs[n]]
            keypoint23 = annotations23[worker23_idxs[n]]

            consolidated_com_all[keypoint1].append(keypoint23)

        consolidated_res[i] = get_mean_annotations_from_dict(consolidated_com_all)
    else:
        consolidated_res[i] = annotations1

# store res, overwrte the previous result
with open('/Users/Evelyn/Desktop/EE148CV/bee-happy/S3_bee_happy_bucket/Annotations/test-keypoint-100demo/consolidated_result_23.json', 'w') as outfile:
    json.dump(consolidated_res, outfile)
'''

'''
consolidated_com = defaultdict(lambda: defaultdict(lambda: []))
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
'''
