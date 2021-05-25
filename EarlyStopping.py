from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn import metrics

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
# model_name = "resnet"
model_name = "vgg"

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 4
# batch_size = 8
# batch_size = 16
# batch_size = 32
# batch_size = 64

# Number of epochs to train for
# num_epochs = 10
# num_epochs = 20
# num_epochs = 50
num_epochs = 60

# Flag for feature extracting.
feature_extract = False #fine tuning
# feature_extract = True #only update the reshaped layer params

#accessing GPU
device = torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

#parameter updates
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

#initialize the model
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# Initialize the model for this run
# model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True) # finetuning
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False) # scratch model - full layer training

# Send the model to GPU
model_ft = model_ft.to(device)

#for grayscale images
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'valid': transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

# Dataset directory path for Real Images classification
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/GANInput_RealData_Filtered/' # 600 N + 600 G

# Dataset directory path for Fake Images classification
data_dir = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered_v2_600/' # 600 N + 600G
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered_v4_5000' # 5000 N + 5000 G
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered_v5_30000' # 30000 N + 30000 G
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered_v6_5000/' # 2nd trial - 5000N 5000G
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered_v7_100000/' # 100000 N + 100000 G

print("Initializing Datasets and Dataloaders...")
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid', 'test']}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'valid', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
print(dataset_sizes)
print('Training image count:',dataset_sizes['train'])
class_names = image_datasets['train'].classes

print('Data directory path : ', data_dir)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9) # SGD
# optimizer_ft = optim.SGD(params_to_update, lr=0.0001, momentum=0.9) # SGD
# optimizer_ft = optim.Adam(params_to_update, lr=0.001, momentum=0.9) # Adam optimizer

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# import EarlyStopping
from pytorchtools import EarlyStopping

# to detect NaN values
# torch.autograd.set_detect_anomaly(True)

patience = 5
# patience = 10

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, patience=5, is_inception=False):
    since = time.time()

    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Early stopping checkpoint for REAL images
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Real_600_Images/60Epoch/earlyStopping_checkpoint.pt'
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Real_600_Images/60Epoch_Patience10/earlyStopping_checkpoint.pt'
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Real_600_Images/60Epoch_BS8_P5/real_es_cp.pt' # batch size 8
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Real_600_Images/60Epoch_BS4_P5/real_es_cp.pt' #batch size - 4
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Real_600_Images/60Epoch_BS4_P10/real_es_cp.pt'

    # Early stopping checkpoint for FAKE images
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_600_Images/10Epoch/earlyStopping_checkpoint.pt'
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_600_Images/60Epoch_patience10/earlyStopping_checkpoint.pt'
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_5000_Images/50Epoch/earlyStopping_checkpoint.pt'
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_30000_Images/50Epoch/earlyStopping_checkpoint.pt'
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_5000_Images/60Epoch_patience10/earlyStopping_checkpoint.pt'

    # Early stopping checkpoint for FAKE images - Synthetic Validation Images
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_600_Images/60Epoch_patience10/Real_Valid/earlyStopping_checkpoint.pt'
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_5000_Images/60Epoch_Patience5/Fake_Valid/earlyStopping_checkpoint.pt'
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_5000_Images/60Ep_BatchSize16_patience5/Real_Valid/real_es_cp.pt' #patience -5
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_5000_Images/60Ep_BatchSize16_patience10/Real_Valid/real_es_cp.pt' #patience -10
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_5000_Images/60Ep_BatchSize16_patience5/Fake_Valid/fake_es_cp.pt' #patience -5
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_30000_Images/60Epoch_patience5/Real_Valid/es_cp.pt' # batch size - 4
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_30000_Images/60Epoch_patience5/Fake_Valid/fake_es_cp.pt' # batch size - 4
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_30000_Images/60Ep_BatchSize16_patience5/Real_Valid/real_es_cp.pt'
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_30000_Images/60Ep_BatchSize16_patience5/Fake_Valid/fake_es_cp.pt'
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_30000_Images/60Ep_BatchSize32_patience5/Fake_Valid/fake_es_cp.pt'
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_30000_Images/60Ep_BatchSize32_patience5/Real_Valid/real_es_cp.pt'

    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_5000_Images_v2/60Epoch_BS16_P5/Real_Valid/real_es_cp.pt'
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_5000_Images_v2/60Epoch_BS16_P5/Fake_Valid/fake_es_cp.pt'

    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_600_Images/60Epoch_BS4_P10/Real_Valid/real_es_cp.pt'
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_600_Images/60Epoch_BS4_P10/Fake_Valid/fake_es_cp.pt'
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_600_Images/60Epoch_BS4_P10_v2/Real_Valid/real_es_cp.pt' # v2 - real of above
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_600_Images/60Epoch_BS4_P10_v2/Fake_Valid/fake_es_cp.pt'
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_600_Images/60Epoch_BS4_P10_v3/Fake_Valid/fake_es_cp.pt'
    earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_600_Images/60Epoch_BS4_P5_v4/Fake_Valid/fake_es_cp.pt'

    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_100000_Images/60Epoch_BS16_P10/Real_Valid/real_es_cp.pt' # batch size 16
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_100000_Images/60Epoch_BS16_P10/Fake_Valid/fake_es_cp.pt' # batch size 16
    # earlyStopping_checkPointPath = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_100000_Images/60Epoch_BS64_P10/Real_Valid/real_es_cp.pt' # batch size 64

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=earlyStopping_checkPointPath)

    try:
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                    print('Training phase')
                else:
                    model.eval()   # Set model to evaluate mode
                    print('Validation phase')

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
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                if phase == 'valid':
                    val_acc_history.append(epoch_acc)
                    val_loss_history.append(epoch_loss)
                if phase == 'train':
                    train_acc_history.append(epoch_acc)
                    train_loss_history.append(epoch_loss)

                if phase == 'valid':
                    # early_stopping needs the validation loss to check if it has decresed,
                    # and if it has, it will make a checkpoint of the current model
                    early_stopping(epoch_loss, model)

                    if early_stopping.early_stop:
                        print("Early stopping")
                        raise StopIteration
                        # break

                # deep copy the model - not in the Early stopping github code
                # below code is from official Pytorch image classification
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print('best model epoch', epoch)
    except StopIteration:
        pass

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # print saved path of early stopping checkpoint model
    print('Early stopping checpoint model is saved at : ', earlyStopping_checkPointPath)

    # # load best model weights
    # model.load_state_dict(best_model_wts)

    # load the last checkpoint with the best model
    # model.load_state_dict(torch.load(earlyStopping_checkPointPath)) # for checkpoints with model weights and parameters
    model = torch.load(earlyStopping_checkPointPath) # for loading model as a whole
        # - Is it actually needed to laod model again?
        # - Maybe model loading can be done just when doing inference
        # - in this case trained model should work - which gets saved inside earlystopping __call__ function

    return model, train_acc_history, train_loss_history, val_acc_history, val_loss_history

# Train and evaluate
model_ft, train_accuracy, train_loss, valid_accuracy, valid_loss = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs,patience=patience, is_inception=(model_name=="inception"))

# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')
# find position of lowest validation loss
minposs = valid_loss.index(min(valid_loss))+1
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
plt.xlabel('epochs')
plt.ylabel('loss')
# plt.ylim(0, 1) # consistent scale
plt.xlim(0, len(train_loss)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('loss_plot.png', bbox_inches='tight')

# # visualize the accuracy as the network trained
# fig = plt.figure(figsize=(10,8))
# plt.plot(range(1,len(train_accuracy)+1),train_accuracy, label='Training Loss')
# plt.plot(range(1,len(valid_accuracy)+1),valid_accuracy,label='Validation Loss')
# # find position of lowest validation loss
# maxacc = valid_accuracy.index(max(valid_accuracy))+1
# plt.axvline(maxacc, linestyle='--', color='r',label='Early Stopping Checkpoint')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.ylim(0, 0.5) # consistent scale
# plt.xlim(0, len(train_accuracy)+1) # consistent scale
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
# fig.savefig('accuracy_plot.png', bbox_inches='tight')





