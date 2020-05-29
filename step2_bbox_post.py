'''
This helps
1. store all consolidated results into one single json file
2. clean the images that faild to label
'''
import json

import os
from collections import defaultdict
from pathlib import Path
from shutil import copyfile

# store all data into same json file

# set the path to the downloaded data:
dir_path = Path('../bee-happy-bucket/Annotations/Pollen/Bee-Happy-Pollen/annotations')
#Path('../bee-happy-bucket/Annotations/Step2/Bee-Happy-final-Step2/annotations')
#

#dir_path = Path('./bee-happy-bucket/Annotations/Step2')

response_path = dir_path/'consolidated-annotation/consolidation-response/iteration-1'
# get sorted list of files:
file_names = sorted(os.listdir(response_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.json' in f]
data = []

for file_name in file_names:
    for line in open(Path(response_path) / file_name, 'r'):
        line_output = json.loads(line)
        #print(line_output)
        for response in line_output:
            annotations = response['consolidatedAnnotation']['content']['Bee-Happy-Pollen']['annotations']#'Bee-Lucky-Step2'
            #print(annotations)
            if len(annotations) < 1:
                continue
            object_values = response['consolidatedAnnotation']['content']['Bee-Happy-Pollen-metadata']['objects']

            invalid_indices = []
            for i,d in enumerate(object_values):
                if d['confidence'] < 0.0:
                    invalid_indices.append(i)
            if len(invalid_indices) == len(annotations):
                continue
            # remove chinese
            #print(response)
            data.append(response)
            #print('DATA:{}'.format(len(data)))

dataset_path = Path('../bee-happy-bucket/caltech-bee') #Path('../bee-lucky-one-video/caltech-bee-one-video')#
bboxes_path = dataset_path/'bboxes_pollen'
images_path = dataset_path/'images_pollen'
# store res, overwrte the previous result
with open(bboxes_path/'all-annotations_pollen.json', 'w') as outfile:
    json.dump(data, outfile)

'''
dealing with the mapping id to image_name
'''
map_id2filename = defaultdict(lambda:'')
annotations_path = dir_path/'intermediate/1/annotations.manifest'

for line in open(annotations_path,'r'):
    line = line.split()
    map_id2filename[line[0]] = line[1][38:]#[37:] #map_id2filename[id] = filename


# copy the succesful labeled images
data_path = Path('../bee-happy-bucket/Datasets/Pollen')#Path('../bee-lucky-one-video/Datasets/Step2')#
for image in data:
    id = image['datasetObjectId']
    file_name = map_id2filename[id]
    #print(file_name)
    copyfile(data_path/file_name,images_path/file_name)

print('Images are copied to' + str(images_path))
