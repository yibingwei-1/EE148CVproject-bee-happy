'''

'''

import json
import os

import torch
import torch.utils.data as data
from PIL import Image,ImageDraw, ImageFont
from collections import defaultdict, OrderedDict


def convert_bbox(width, top, height, left):
    '''
    the ground-truth boxes in [x1, y1, x2, y2] format, with values of x
          between 0 and W and values of y between 0 and H
    '''
    return [left, top, left + width, top + height]


def check_box_size(width, height, threshold=20):
    if width > threshold and height > threshold:
        return True
    else:
        return False

'''
visualize the box to check whether images match the bboxes
'''
def visualize(im,labels,boxes):
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("./calibril.ttf", 15)

    colors = ['black','blue','orange']
    texts = ['bkgrd','normal bee','pollen bee']
    for i in range(len(labels)):
        label = labels[i]
        box_location = boxes[i]#[a['left'],a['top'],a['left']+a['width'],a['top']+a['height']]
        draw.rectangle(xy=box_location, outline=colors[label],width=5) #[x0, y0, x1, y1]

        # Text
        text_size = font.getsize(texts[label].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=colors[label])
        draw.text(xy=text_location, text=texts[label].upper(), fill='white',
                  font=font)

    del draw
    im.show()


class caltech_bee(data.Dataset):
    '''
    A map-style dataset is one that implements the __getitem__() and __len__() protocols, and represents a map from (possibly non-integral) indices/keys to data samples.

    For example, such a dataset, when accessed with dataset[idx], could read the idx-th image and its corresponding label from a folder on the disk.
    '''

    def __init__(self, root='../bee-happy-bucket/caltech-bee', transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned

        imgs = list(sorted(os.listdir(os.path.join(root, "images_ignoreNoLabel"))))
        self.imgs = [f for f in imgs if '.jpg' in f]
        print('Img:{}'.format(len(self.imgs)))

        # get bounding box coordinates for each mask
        bbox_path = os.path.join(self.root, "bboxes", 'all-annotations_ignoreNoLabel.json')

        for line in open(bbox_path, 'r'):
            responses = json.loads(line)  # one line

        temp_dict = {i: None for i in range(1000)}
        for response in responses:
            id = int(response['datasetObjectId'])
            temp_dict[id] = response

        self.bboxes = []
        for response in temp_dict.values():
            if not response:
                continue
            self.bboxes.append(response)

        print('BBOX:{}'.format(len(self.bboxes)))

        # self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        # self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and bbox
        img_path = os.path.join(self.root, "images_ignoreNoLabel", self.imgs[idx])

        img = Image.open(img_path).convert("RGB")

        consolidated_response = self.bboxes[idx]
        # map_idx2label = consolidated_response['consolidatedAnnotation']['content']['Bee-Happy-final-Step2-metadata']['class-map']
        object_values = consolidated_response['consolidatedAnnotation']['content']['Bee-Happy-final-Step2-metadata']['objects'] #list of dict [{'confidence':0.24}]
        valid_indices = []
        for i,d in enumerate(object_values):
            if d['confidence'] >= 0.1:
                valid_indices.append(i)

        annotations = consolidated_response['consolidatedAnnotation']['content']['Bee-Happy-final-Step2']['annotations']
        num_objs = len(annotations)

        labels = []
        boxes = []
        for idx in valid_indices:
            # to remove label 0 (refer to background)
            # 1: pollen
            # 2: without pollen
            a = annotations[idx]
            if not check_box_size(a['height'], a['width']):
                continue
            labels.append(int(a['class_id']) + 1)
            boxes.append(convert_bbox(a['width'], a['top'], a['height'], a['left']))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

# data = caltech_bee(root='./bee-happy-bucket/caltech-bee', transforms=None)
# for i in [7]:
#     data.__getitem__(i)
