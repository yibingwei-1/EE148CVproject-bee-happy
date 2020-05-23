'''
plot points on images to even out the worklaod
'''
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

anno_dir_path = Path('../bee-happy-bucket/Annotations/Step1/Bee-Happy-True-Step1/')


def get_circle_bbox(center, r=10):
    # return [x0, y0, x1, y1]
    x, y = center
    return [x - r, y - r, x + r, y + r]


'''
dealing with the mapping id to image_name
'''
map_id2filename = defaultdict(lambda: '')
annotations_path = anno_dir_path / 'annotations/intermediate/1/annotations.json'
for line in open(annotations_path, 'r'):
    line = json.loads(line)
    map_id2filename[str(line['datasetObjectId'])] = line["inputAttribute"][37:]

# load consolidated_result.json
json_path = anno_dir_path/'annotations/consolidated-annotations.json'
for line in open(json_path, 'r'):
    consolidated_data = json.loads(line)


dataset_dir_path = Path('../bee-happy-bucket/Datasets/Step1')
save_dir_path = Path('../bee-happy-bucket/Datasets/Step2')
for i in range(len(consolidated_data)):
    datasetObjectId = str(i)
    # plot consolidated result
    im = Image.open(dataset_dir_path/ map_id2filename[datasetObjectId])
    im_draw = im.copy()
    draw = ImageDraw.Draw(im_draw)

    # if there is no bee in the image of datasetObjectId
    if datasetObjectId not in consolidated_data:
        continue
    # Now, loop through coord arrays, and create a circle at each x,y pair
    count = 0
    for count, keypoint in enumerate(consolidated_data[datasetObjectId]):
        #     draw.ellipse(get_circle_bbox(keypoint[0]), fill='yellow')
        #
        # im_draw.save(dir_path/'Datasets/test-keypoint-100demo-annotated/annotated_id{}_{}.jpg'.format(i,count+1))

        if count % 8 == 0 and count != 0:
            im_draw.save(save_dir_path / 'annotated_id{:03d}_{:02d}.jpg'.format(i, count))
            im_draw = im.copy()
            draw = ImageDraw.Draw(im_draw)

        draw.ellipse(get_circle_bbox(keypoint[0]), fill='yellow')

    im_draw.save(save_dir_path / 'annotated_id{:03d}_{:02d}.jpg'.format(i, count+1))
