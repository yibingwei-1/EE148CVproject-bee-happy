import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset, get_coco_api_from_dataset2
from coco_eval import CocoEvaluator
import utils

import torch.nn.functional as F


def train(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        # Tensor.to() Returns a Tensor with the specified device and (optional) dtype
        # A torch.device is an object representing the device on which a torch.Tensor is or will be allocated.
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger, losses_reduced/len(data_loader)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        # Tensor.to() Returns a Tensor with the specified device and (optional) dtype
        # A torch.device is an object representing the device on which a torch.Tensor is or will be allocated.
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset2(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        predictions = model(images)

        predictions = [{k: v.to(cpu_device) for k, v in t.items()} for t in predictions] #[{'boxes':,'labels':,'scores':}] batch_size = 1

        outputs = []
        count = 0
        for image in predictions:
            boxes = image['boxes']
            scores = image['scores']
            lables = image['labels']
            indices = torch.ops.torchvision.nms(boxes, scores, 0.1)
            count += 1

            boxes = torch.stack([boxes[i] for i in indices])
            scores = torch.stack([scores[i] for i in indices])
            lables = torch.stack([lables[i] for i in indices])
            outputs.append({'boxes':boxes,'labels':lables,'scores':scores})

        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

from PIL import ImageDraw, ImageFont
import os

def visualize(model, data_loader, device, epoch, save_folder, gt = False):

    model.eval()
    cpu_device = torch.device("cpu")

    count = 0
    print('Visualize:{}'.format(len(data_loader)))
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        predictions = model(images)

        predictions = [{k: v.to(cpu_device) for k, v in t.items()} for t in predictions] #[{'boxes':,'labels':,'scores':}] batch_size = 1
        outputs = []
        for image in predictions:
            boxes = image['boxes']
            scores = image['scores']
            lables = image['labels']
            indices = torch.ops.torchvision.nms(boxes, scores, 0.1)

            boxes = torch.stack([boxes[i] for i in indices])
            scores = torch.stack([scores[i] for i in indices])
            lables = torch.stack([lables[i] for i in indices])
            outputs.append({'boxes':boxes,'labels':lables,'scores':scores})
        if gt: # plot ground truth for debug
            for image, output, target in zip(images,outputs,targets):
                annotated_image = _visualize_one_image_wTarget(image,output,target)
                annotated_image.save(os.path.join(save_folder, '{}_epoch{}_wTarget.jpg'.format(count,epoch)))
                count += 1
        else:
            for image, output in zip(images,outputs):
                annotated_image = _visualize_one_image(image,output)
                annotated_image.save(os.path.join(save_folder, '{}_epoch{}.jpg'.format(count,epoch)))
                count += 1

import torchvision.transforms as transforms

def _visualize_one_image(image,output):
    '''

    :param image: Tensor (C,H,W)
    :param target: dictionary
    :return:
    '''
    cpu_device = torch.device("cpu")
    im = transforms.ToPILImage(mode='RGB')(image.to(cpu_device))
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("./calibril.ttf", 15)

    boxes = output["boxes"].tolist()
    labels = output["labels"].tolist()
    colors = ['black','blue','orange']
    texts = ['bkgrd','normal bee','pollen bee']
    for i,label in enumerate(labels):
        box_location = boxes[i]
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
    return im

def _visualize_one_image_wTarget(image,output,target):
    '''

    :param image: Tensor (C,H,W)
    :param target: dictionary
    :return:
    '''
    cpu_device = torch.device("cpu")
    im = transforms.ToPILImage(mode='RGB')(image.to(cpu_device))
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("./calibril.ttf", 15)

    boxes = output["boxes"].tolist()
    gts = target["boxes"].tolist()

    labels = output["labels"].tolist()
    colors = ['black','blue','orange']

    texts = ['bkgrd','normal bee','pollen bee']
    for i in range(len(gts)):
        # gt
        gt_location = gts[i]
        draw.rectangle(xy=gt_location, outline='green',width=5) #[x0, y0, x1, y1]


    for i,label in enumerate(labels):


        # prediction
        box_location = boxes[i]
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
    return im

# def _visualize_one_image(image,target):
#     '''
#
#     :param image: Tensor (C,H,W)
#     :param target: dictionary
#     :return:
#     '''
#     cpu_device = torch.device("cpu")
#     im = transforms.ToPILImage(mode='RGB')(image.to(cpu_device))
#     draw = ImageDraw.Draw(im)
#     font = ImageFont.truetype("./calibril.ttf", 15)
#
#     boxes = target["boxes"].tolist()
#     labels = target["labels"].tolist()
#     colors = ['black','blue','orange']
#     texts = ['bkgrd','normal bee','pollen bee']
#     for i,label in enumerate(labels):
#         box_location = boxes[i]
#         draw.rectangle(xy=box_location, outline=colors[label],width=5) #[x0, y0, x1, y1]
#
#         # Text
#         text_size = font.getsize(texts[label].upper())
#         text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
#         textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
#                             box_location[1]]
#         draw.rectangle(xy=textbox_location, fill=colors[label])
#         draw.text(xy=text_location, text=texts[label].upper(), fill='white',
#                   font=font)
#
#     del draw
#     return im



