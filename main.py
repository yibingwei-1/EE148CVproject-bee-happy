from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


from engine import train_one_epoch, evaluate
import utils
import transforms as T

import torch, torchvision


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
dataset = caltech_bee(transforms=get_transform(train=True))
dataset_test = caltech_bee(transforms=get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist() # random permuation -> list

dataset_train = torch.utils.data.Subset(dataset, indices[:-200])
print('Training data size:{}'.format(len(dataset_train)))
dataset_validation = torch.utils.data.Subset(dataset, indices[-200:-100])

dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=2, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)

data_loader_validation = torch.utils.data.DataLoader(
    dataset_validation, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)
'''
collate_fn (callable, optional) – merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset.
the collate_fn is your callable/function that processes the batch you want to return from your dataloader.
the collate_fn receives a list of tuples if your __getitem__ function from a Dataset subclass returns a tuple, or just a normal list if your Dataset subclass returns only one element.
If you don’t use it, PyTorch only put batch_size examples together as you would using torch.stack 
https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3 
'''
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 3

# get the model using our helper function
model = get_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params)
# torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# # and a learning rate scheduler which decreases the learning rate by
# # 10x every 3 epochs
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                step_size=3,
#                                                gamma=0.1)

# let's train it for 10 epochs
num_epochs = 200

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
    # # update the learning rate
    # lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_validation, device=device)

# test on testset

evaluate(model,data_loader_test, device=device)
