import matplotlib.pyplot as plt
import json
from matplotlib.patches import Circle

from collections import defaultdict
from pathlib import Path

from PIL import  Image, ImageDraw

dir_path = Path('/Users/Evelyn/Desktop/EE148CV/bee-happy/S3_bee_happy_bucket')

def get_circle_bbox(center,r=6):
    #return [x0, y0, x1, y1]
    x,y = center
    return [x-r,y-r,x+r,y+r]

'''
dealing with the mapping id to image_name
'''
map_id2filename = defaultdict(lambda:'')
annotations_path = dir_path/'Annotations/test-keypoint-100demo/annotations/intermediate/1/annotations.json'
for line in open(annotations_path,'r'):
    line = json.loads(line)
    map_id2filename[str(line['datasetObjectId'])] = line["inputAttribute"][40:]


# load consolidated_result.json
json_path = dir_path/'Annotations/test-keypoint-100demo/consolidated_result.json'
for line in open(json_path,'r'):
    consolidated_data = json.loads(line)

for i in range(len(consolidated_data)):
    datasetObjectId = str(i)
    # plot consolidated result
    im = Image.open(dir_path/"Datasets/test-keypoint-100demo/"/map_id2filename[datasetObjectId])
    draw = ImageDraw.Draw(im)

    # Now, loop through coord arrays, and create a circle at each x,y pair
    for keypoint in consolidated_data[datasetObjectId]:

        draw.ellipse(get_circle_bbox(keypoint[0]), fill='yellow')

    im.save(dir_path/'Datasets/test-keypoint-100demo-annotated/annotated_{}.jpg'.format(i))

