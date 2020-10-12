import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from torch.optim import lr_scheduler
from sklearn import metrics
from collections import Counter

# data_dir = '/home/sysadmin/Ashish/Chinese_Splitdata'
# data_dir = '/home/sysadmin/Ashish/Chinese_Fake_Glaucoma_ClassificationData'
data_dir = '/home/sysadmin/Ashish/All_Classification'

batch_size = 30
# def load_split_train_test(datadir, valid_size = .2, test_size = 0.2):
#     train_transforms = transforms.Compose([transforms.Resize((224,224)),
#                                            transforms.ToTensor(),
#                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                                            ])
#     valid_transforms = transforms.Compose([transforms.Resize((224,224)),
#                                            transforms.ToTensor(),
#                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                                            ])
#     test_transforms = transforms.Compose([transforms.Resize((224,224)),
#                                            transforms.ToTensor(),
#                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                                            ])
#     train_data = datasets.ImageFolder(datadir,
#                     transform=train_transforms)
#     valid_data = datasets.ImageFolder(datadir,
#                     transform=valid_transforms)
#     test_data = datasets.ImageFolder(datadir,
#                     transform=test_transforms)
#
#     num_train = len(train_data)
#     indices = list(range(num_train))
#     valid_split = int(np.floor(valid_size * num_train))
#     test_split = int(np.floor(test_size * num_train))
#     np.random.shuffle(indices)
#
#     from torch.utils.data.sampler import SubsetRandomSampler
#     test_idx, valid_idx, train_idx = indices[:test_split], indices[test_split:(valid_split+test_split)], indices[(valid_split+test_split):]
#
#     train_sampler = SubsetRandomSampler(train_idx)
#     valid_sampler = SubsetRandomSampler(valid_idx)
#     test_sampler = SubsetRandomSampler(test_idx)
#
#     trainloader = torch.utils.data.DataLoader(train_data,
#                    sampler=train_sampler, batch_size=batch_size)
#     validloader = torch.utils.data.DataLoader(valid_data,
#                    sampler=valid_sampler, batch_size=batch_size)
#     testloader = torch.utils.data.DataLoader(test_data,
#                    sampler=test_sampler, batch_size=batch_size)
#     return trainloader, validloader, testloader, train_idx, valid_idx, test_idx
#
# trainLoader, validLoader, testLoader, train_idx, valid_idx, test_idx = load_split_train_test(data_dir,0.2,0.1)
# print('train dataloader iterations : ',len(trainLoader), 'batch size : ', batch_size)
# print('valid dataloader iterations : ',len(validLoader),'batch size : ', batch_size)
# print('test dataloader iterations : ',len(testLoader),'batch size : ', batch_size)
# dataloaders = {
#     'train' : trainLoader,
#     'valid' : validLoader,
#     'test' : testLoader
# }
# dataset_sizes = {
#     'train' : len(train_idx),
#     'valid' : len(valid_idx),
#     'test' : len(test_idx)
# }

def load_split_train_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                           ])
    valid_transforms = transforms.Compose([transforms.Resize((224,224)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                           ])

    train_data = datasets.ImageFolder(datadir,
                    transform=train_transforms)
    valid_data = datasets.ImageFolder(datadir,
                    transform=valid_transforms)


    num_train = len(train_data)
    indices = list(range(num_train))
    valid_split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)

    from torch.utils.data.sampler import SubsetRandomSampler
    valid_idx, train_idx = indices[:valid_split],indices[valid_split:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=batch_size)
    validloader = torch.utils.data.DataLoader(valid_data,
                   sampler=valid_sampler, batch_size=batch_size)

    return trainloader, validloader, train_idx, valid_idx

trainLoader, validLoader, train_idx, valid_idx = load_split_train_test(data_dir,0.2)
print('train dataloader iterations : ',len(trainLoader), 'batch size : ', batch_size)
print('valid dataloader iterations : ',len(validLoader),'batch size : ', batch_size)


#
# count = 0
# images_so_far = 0
# for images, labels in trainLoader:
#     for i in range(images.size()[0]):
#         images_so_far +=1
# print(images_so_far)

dataloaders = {
    'train' : trainLoader,
    'valid' : validLoader,
}
dataset_sizes = {
    'train' : len(train_idx),
    'valid' : len(valid_idx),
}
class_names = trainLoader.dataset.classes
print(dataset_sizes)
# print(trainLoader.dataset.class_to_idx)
print(dict(Counter(trainLoader.dataset.targets)))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
inputs, classes = next(iter(trainLoader))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

# imshow(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
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

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        # for i, (inputs, labels) in enumerate(dataloaders['test']): // if using 'test' during dataloader splitting of data
        for i, (inputs, labels) in enumerate(dataloaders['valid']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true += list(labels.cpu().numpy())
            y_pred += list(preds.cpu().numpy())
            total += labels.size(0)
            correct += (preds == labels.data).sum().item()

            # for j in range(inputs.size()[0]):
            #     images_so_far += 1
                # ax = plt.subplot(num_images//2, 2, images_so_far)
                # ax.axis('off')
                # ax.set_title('pred:{} org:{}'.format(class_names[preds[j]], class_names[labels[j]]))
                # imshow(inputs.cpu().data[j])

                # if images_so_far == num_images:
                #     model.train(mode=was_training)
                #     return
        model.train(mode=was_training)
        print('Test accuracy using Pytorch official doc method : {:0.2f} %'.format(100 * (correct / total)))

        # evaluation metrics
        print(metrics.accuracy_score(y_true, y_pred))
        from sklearn.metrics import classification_report
        print(classification_report(y_true, y_pred))
        print('finished!!!')


# model_ft = models.resnet18(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# # Here the size of each output sample is set to 2.
# # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = models.vgg16(pretrained=True)
num_ftrs = model_ft.classifier[6].in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft.classifier[6] = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)
# model_ft = nn.DataParallel(model_ft)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=30)
# Specify a path
PATH = "All_Classification_model_v2.pt"
# Save
torch.save(model_ft, PATH)

visualize_model(model_ft)
