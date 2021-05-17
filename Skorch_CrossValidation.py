import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import os
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
import pandas as pd

data_dir = '/home/sysadmin/Ashish/Chinese_Classification_CrossValidation'

train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                       ])
valid_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                       ])

# train_data = datasets.ImageFolder(data_dir,
#                                   transform=train_transforms)
# valid_data = datasets.ImageFolder(data_dir,
#                                   transform=valid_transforms)

train_ds = datasets.ImageFolder(
    os.path.join(data_dir, 'train'), train_transforms)
val_ds = datasets.ImageFolder(
    os.path.join(data_dir, 'valid'), valid_transforms)

class PretrainedModel(nn.Module):
    def __init__(self, output_features):
        super().__init__()
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_features)
        self.model = model

    def forward(self, x):
        return self.model(x)

from skorch.callbacks import LRScheduler
lrscheduler = LRScheduler(
    policy='StepLR', step_size=7, gamma=0.1)

from skorch.callbacks import Checkpoint
checkpoint = Checkpoint(
    f_params='best_model.pt', monitor='valid_acc_best')

from skorch.callbacks import Freezer
freezer = Freezer(lambda x: not x.startswith('model.fc'))

net = NeuralNetClassifier(
    PretrainedModel,
    criterion=nn.CrossEntropyLoss,
    lr=0.001,
    batch_size=4,
    max_epochs=25,
    module__output_features=2,
    optimizer=optim.SGD,
    optimizer__momentum=0.9,
    iterator_train__shuffle=True,
    iterator_train__num_workers=4,
    iterator_valid__shuffle=True,
    iterator_valid__num_workers=4,
    train_split=predefined_split(val_ds),
    # train_split=None,
    callbacks=[lrscheduler, checkpoint, freezer],
    device='cuda' # comment to train on cpu
)
net.fit(train_ds, y=None)

test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                       ])
test_ds = datasets.ImageFolder(
    os.path.join(data_dir, 'valid'), test_transforms)

net.predict(train_ds)
print('')
