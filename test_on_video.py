'''
Test the detection performance of trained network on a video
'''
# Object Detection

import imageio
# Importing the libraries
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import ImageDraw, ImageFont
import torchvision.transforms as transforms
import os
from torch.autograd import Variable
import numpy as np

# Defining a function that will do the detections
def detect(frame, model):
    # We define a detect function that will take as inputs, a frame, a ssd neural network, and a transformation to be applied on the images, and that will return the frame with the detector rectangle.

    cpu_device = torch.device("cpu")

    predictions = model(frame)  # We feed the neural network ssd with the image and we get the output y.
    predictions = [{k: v.to(cpu_device) for k, v in t.items()} for t in
                   predictions]  # [{'boxes':,'labels':,'scores':}] batch_size = 1

    image = predictions[0]
    boxes = image['boxes']
    scores = image['scores']
    lables = image['labels']
    indices = torch.ops.torchvision.nms(boxes, scores, 0.1)

    boxes = torch.stack([boxes[i] for i in indices])
    scores = torch.stack([scores[i] for i in indices])
    lables = torch.stack([lables[i] for i in indices])
    output = {'boxes': boxes, 'labels': lables, 'scores': scores}

    return _visualize_one_image(frame, output)

def _visualize_one_image(image, output):
    '''

    :param image: Tensor (C,H,W)
    :param target: dictionary
    :return:
    '''
    cpu_device = torch.device("cpu")
    image = image.squeeze(0)
    im = transforms.ToPILImage(mode='RGB')(image.to(cpu_device))
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("./calibril.ttf", 15)

    boxes = output["boxes"].tolist()
    labels = output["labels"].tolist()
    colors = ['black', 'blue', 'orange']
    texts = ['bkgrd', 'normal bee', 'pollen bee']
    for i, label in enumerate(labels):
        box_location = boxes[i]
        draw.rectangle(xy=box_location, outline=colors[label], width=5)  # [x0, y0, x1, y1]

        # Text
        text_size = font.getsize(texts[label].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=colors[label])
        draw.text(xy=text_location, text=texts[label].upper(), fill='white',
                  font=font)

    del draw
    return im

def generate_detection_video(input_video, output_video):
    reader = imageio.get_reader(input_video)  # We open the video.
    fps = reader.get_meta_data()['fps']  # We get the fps frequence (frames per second).
    writer = imageio.get_writer(output_video, fps=fps)  # We create an output video with this same fps frequence.
    for i, frame in enumerate(reader):  # We iterate on the frames of the output video:
        frame = loader(frame).float()
        frame = Variable(frame)
        frame = frame.unsqueeze(0)
        frame = detect(frame, model.eval())  # We call our detect function (defined above) to detect the object on the frame.
        frame = np.asarray(frame)
        writer.append_data(frame)  # We add the next frame in the output video.
        print(input_video+'-frame:'+str(i))  # We print the number of the processed frame.
    writer.close()  # We close the process that handles the creation of the output video.

# load the trained model
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280

anchor_generator = AnchorGenerator(sizes=((64, 128, 256),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)
# num_classes: background, normal, pollen
model = FasterRCNN(backbone, num_classes=3, rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model_path = './model_mixed_epoch20.pt'
model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))


loader = transforms.Compose([transforms.ToTensor()])

# Doing some Object Detection on videos
video_dir = './videos'
file_names = sorted(os.listdir(video_dir))
file_names = [f for f in file_names if '.MOV' in f]

for input in file_names:
    output = 'detection_' + input
    input_video = os.path.join(video_dir, input)
    output_video = os.path.join(video_dir, output)
    generate_detection_video(input_video, output_video)


