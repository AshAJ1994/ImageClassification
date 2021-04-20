from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torchviz import make_dot

# Data augmentation and normalization for training
# Just normalization for validation
# for coloured images
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'valid': transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'test': transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }

# for grayscale images
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

# data_dir = '/home/sysadmin/Ashish/Chinese_Glaucoma_ClassificationData'
# data_dir = '/home/sysadmin/Ashish/Oct21/Chinese_original_images'
# data_dir = '/home/sysadmin/Ashish/Oct21/Chinese_fake_images'

# data_dir = '/home/sysadmin/Ashish/40000_Classification'
# data_dir = '/home/sysadmin/Ashish/All_Classification'
# data_dir = '/home/sysadmin/Ashish/latest_GAN_v2.0'
# data_dir = '/home/sysadmin/Ashish/Mixed(GAN+Real)'
# data_dir = '/home/sysadmin/Ashish/latestGAN_v3.0_100000'

#validation from scratch
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/40000FakeImages_training'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/40000FakeImages_training/Models/PytorchMethod2(Resnet)/'

# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/40000FakeImages+realNormalImages_training'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/40000FakeImages+realNormalImages_training/Models/PytorchMethod2(Resnet)/'

# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/50000FakeImages+realImages_traiining'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/50000FakeImages+realImages_traiining/Models/PytorchMethod2(Resnet)/'

# real images classification - 900 train + 100 valid + 97 test images data split for normal and glaucoma
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/RealImages(1590,523+627)_Training'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/RealImages(1590,523+627)_Training/Models/PytorchMethod2(Resnet)/'
#
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/50000Fake+900RealImages_Training'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/50000Fake+900RealImages_Training/Models/PytorchMethod2(Resnet)/'

# 50000 fake images only (from latest GAN method for Nov4th meeting : without test images)-- no real images included for classification
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/50000_FakeImagesOnly_training'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/50000_FakeImagesOnly_training/Models/PytorchMethod2(Resnet)'

# Finalized GAN - Real Images - 1100 training + 100 valid + 72 test : RNFL based data split for Glaucoma
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA/Models/PytorchMethod2(Resnet)/'

# # Finalized GAN - Fake Images - 50000G + 50000 N ( from 1100 G + 1100 N training: RNFL based data split for Glaucoma)
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/FinalizedGAN_Severity_GANImages'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/FinalizedGAN_Severity_GANImages/Models/ResnetOnly'

# Finalized GAN - Fake Images - 50000G + 50000 N ( from 1000 G + 1000 N training, 10000 N + 5000G - valid: Severity based data split for Glaucoma)
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/FInalized_GAN_basedOnSeverity_FakeOnly'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/FInalized_GAN_basedOnSeverity_FakeOnly/Models/ResnetOnly/'

# Finalized GAN - Real Images - 1000 training + 100 valid + 78 test : Severity based data split for Glaucoma
data_dir = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/Models/PytorchMethod2(Resnet)/'

# ants bees dataset
# data_dir = '/home/sysadmin/Ashish/ants_bees_dataset'
# modelSavingPath = '/home/sysadmin/Ashish/ants_bees_dataset/Models/Resnet/' - for coloured images
# modelSavingPath = '/home/sysadmin/Ashish/ants_bees_dataset/Models/BW_Resnet/' # - for grayscale images

#for CAM visualization - trained on Real Images - Grayscale
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/Models/CAM/B&W/'

#for CAM visualization - trained on FLATTENED Real Images - Grayscale
data_dir = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_Severity_FlattemedImages/'
modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_Severity_FlattemedImages/Models/CAM/B&W/'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'valid', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
print(dataset_sizes)
class_names = image_datasets['train'].classes

# device = torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch_number = ''

    imgsSoFar = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:

                imgsSoFar += 1

                inputs = inputs.to(device)
                labels = labels.to(device)

                # visualizeFeatureMap(model, inputs)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # make_dot(outputs)

                    # print(outputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch_number = str(epoch)


        print(imgsSoFar)
        if (epoch%2 == 0):
            # progressPath = '1lkhImage_v2.0_'+str(epoch)+'.pt'
            progressPath = modelSavingPath+'Resnet'+'_'+str(dataset_sizes['train'])+'_'+str(epoch)+'.pt'
            torch.save(model, progressPath)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best epoch is : ', best_epoch_number)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Visualizing feature map
# https://androidkt.com/how-to-visualize-feature-maps-in-convolutional-neural-networks-using-pytorch/
def visualizeFeatureMap(model, img):
    no_of_layers = 0
    conv_layers = []

    model_children = list(model.children())

    for child in model_children:
        if type(child) == nn.Conv2d:
            no_of_layers += 1
            conv_layers.append(child)
        elif type(child) == nn.Sequential:
            # print(child.children())
            for layer in child.children():
                for eachBlock in layer.children():
                    # print(eachBlock)
                    if type(eachBlock) == nn.Conv2d:
                        no_of_layers += 1
                        conv_layers.append(layer)
    print(no_of_layers)

    results = [conv_layers[0](img)]
    for i in range(1, len(conv_layers)):
        results.append(conv_layers[i](results[-1]))
    chk = results


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    # class_names = ('glaucoma', 'normal')
    class_names = ('ants', 'bees')
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)


            # for j in range(inputs.size()[0]):
            #     images_so_far += 1
            #     ax = plt.subplot(num_images//2, 2, images_so_far)
            #     ax.axis('off')
            #     ax.set_title('pred:{} org:{}'.format(class_names[preds[j]], class_names[labels[j]]))
            #     imshow(inputs.cpu().data[j])
            #
            #     if images_so_far == num_images:
            #         model.train(mode=was_training)
            #         return
    #     model.train(mode=was_training)
    #     y_true += list(labels.cpu().numpy())
    #     y_pred += list(preds.cpu().numpy())
    #     total += labels.size(0)
    #     correct += (preds == labels.data).sum().item()
    # print('Test accuracy using Pytorch official doc method : {:0.2f} %'.format(100 * (correct / total)))
    print('done!!!')

model_ft = models.resnet18(pretrained=True)
for param in model_ft.parameters():
    param.require_grad = False
num_ftrs = model_ft.fc.in_features
# # Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

# model_ft = models.vgg16(pretrained=True)
# # Freeze training for all layers
# for param in model_ft.features.parameters():
#     param.require_grad = False
# num_ftrs = model_ft.classifier[6].in_features
# model_ft.classifier[6] = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)
# # model_ft = nn.DataParallel(model_ft)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

num_epochs=10
# num_epochs = 20

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs)
# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
#                        num_epochs=30)
# Specify a path
# PATH = "Oct21_FullData_Orignal_model.pt"
# PATH = 'Oct21_FakeModel.pt'
# PATH = 'vgg_freezed_originalimages.pt'
# PATH = 'vgg_freezed_originalimages_bugFix.pt'
# PATH = 'vgg_freezed_Fakeimages.pt'
# PATH = 'vgg_freezed_Fakeimages_bugFix.pt'
# PATH = 'resnet_freezed_Fakeimages_bugFix.pt'
# PATH = 'resnet_finetune_full_OriginalImages.pt'
# PATH = 'resnet_finetune_full_FakeImages.pt'

# PATH = 'LargeSize_FakeModel_ResNet.pt'
# PATH = 'LargeSize_FakeModel_VGG16.pt'

# PATH = 'latest_Resnet_realimages.pt'
# PATH = 'hope.pt'
# PATH = 'hope_Epoch30.pt'
# PATH = 'Mixed_Resnet_Model.pt'
# PATH = 'Terminal_1lkhImage_v2.0_Resnet.pt'

savedModelName = modelSavingPath +'Resnet'+'_'+str(dataset_sizes['train'])+'_bestEpoch'+'.pt'

# Save
torch.save(model_ft, savedModelName)


visualize_model(model_ft)


print('saved model is : ', savedModelName)