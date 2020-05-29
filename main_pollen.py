import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# load a pre-trained model for classification and return
# only the features
backbone = torchvision.models.mobilenet_v2(pretrained=False).features
# FasterRCNN needs to know the number of
# output channels in a backbone. For mobilenet_v2, it's 1280
# so we need to add it here
backbone.out_channels = 1280

# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and
# aspect ratios
anchor_generator = AnchorGenerator(sizes=((64, 128, 256),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

# put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone,
                   num_classes=3,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

from engine import evaluate, visualize, train
import utils
import transforms as T

import torch


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


from caltech_bee import caltech_bee

# use our dataset and defined transformations
root_path ='../bee-happy-bucket/caltech-bee'
#root_path = '../bee-lucky-one-video/caltech-bee-one-video'
dataset = caltech_bee(root=root_path, transforms=get_transform(train=True))
dataset_validation = caltech_bee(root=root_path, transforms=get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()  # random permuation -> list

validation_rate = 0.15

dataset_train = torch.utils.data.Subset(dataset, indices)  # -200])#int(len(indices)*(1-validation_rate))
print('Training data size:{}'.format(len(dataset_train)))


# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=2, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)


# data_loader_test = torch.utils.data.DataLoader(
#     dataset_test, batch_size=1, shuffle=False, num_workers=0,
#     collate_fn=utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# move model to the right device
model.to(device)

model_path = './model_pollen_epoch10.pt'
model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.Adam(params)
# # torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

import os

# let's train it for 10 epochs
num_epochs = 10
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    #train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
    _, loss = train(model, optimizer, data_loader, device, epoch, print_freq=1)
    train_losses.append(loss)
    # # update the learning rate
    lr_scheduler.step()

import matplotlib.pyplot as plt
# Plot training and val loss as a function of the epoch. Use this to monitor for overfitting.
plt.plot(range(1, num_epochs + 1), train_losses, label='traning')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.savefig(os.path.join(root_path, 'prediction_visualization','loss.png'), bbox_inches='tight')

torch.save(model.state_dict(), "model_pollen_epoch{}.pt".format(num_epochs))
# test on testset

#evaluate(model, data_loader_test, device=device)
