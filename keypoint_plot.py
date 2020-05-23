'''
viusalize the keypoint
'''
import matplotlib.pyplot as plt
import json
from matplotlib.patches import Circle

from collections import defaultdict
from pathlib import Path

dir_path = Path('./bee-happy-bucket')

'''
dealing with the mapping id to image_name
'''
map_id2filename = defaultdict(lambda:'')
annotations_path = dir_path/'Annotations/test-keypoint-100demo/annotations/intermediate/1/annotations.json'
for line in open(annotations_path,'r'):
    line = json.loads(line)
    map_id2filename[str(line['datasetObjectId'])] = line["inputAttribute"][40:]


# load consolidated_result.json
json_path = dir_path/'Annotations/test-keypoint-100demo/consolidated_result_naive.json'
for line in open(json_path,'r'):
    consolidated_data = json.loads(line)

datasetObjectId = '16'
# plot consolidated result
img = plt.imread(dir_path/"Datasets/test-keypoint-100demo/"/map_id2filename[datasetObjectId])

# Create a figure. Equal aspect so circles look circular
fig,ax = plt.subplots(2)
ax[0].set_aspect('equal')
ax[0].imshow(img)

# Now, loop through coord arrays, and create a circle at each x,y pair
for keypoint in consolidated_data[datasetObjectId]:
    x = keypoint[0][0]
    y = keypoint[0][1]
    circ = Circle((x,y),5,color='yellow')
    ax[0].add_patch(circ)


# first read the json
demo_path = './bee-happy-bucket/Annotations/test-keypoint-100demo/test100-all-annotations.json'
data = []
for line in open(demo_path, 'r'):
    data = json.loads(line)

n_images = len(data)


# object ID -> the actual random idx
def worker_responses(data,datasetObjectId):

    for i,image in enumerate(data):
        if image['datasetObjectId']==datasetObjectId:
            return int(i)

    return -1


# load json data
color = ['red','green','blue']
ax[1].set_aspect('equal')
ax[1].imshow(img)

image=data[worker_responses(data,datasetObjectId)]
workers = image["consolidatedAnnotation"]["content"]["test-keypoint-100demo"][
    'annotationsFromAllWorkers']  # list of dictionary [{}], each dict is one work
for j, worker in enumerate(workers):
    annot = eval(worker['annotationData']['content'])
    keypoints = annot['annotatedResult']['keypoints']  # list of dictionary [{}], each dict is one point
    for keypoint in keypoints:
        x = keypoint['x']
        y = keypoint['y']

        circ = Circle((x,y),5,color=color[j])
        ax[1].add_patch(circ)
# Show the image
plt.show()

plt.imsave('res.jpg',img)
