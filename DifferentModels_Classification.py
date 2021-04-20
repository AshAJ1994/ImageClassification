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
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure

# data_dir = "/home/sysadmin/Ashish/All_Classification"
# data_dir = '/home/sysadmin/Ashish/40000_Classification'
# data_dir = '/home/sysadmin/Ashish/20000_Classification'
# data_dir = '/home/sysadmin/Ashish/latest_GAN_generatedImages'
# data_dir = '/home/sysadmin/Ashish/latest_GAN_v2.0'
# data_dir = '/home/sysadmin/Ashish/Mixed(GAN+Real)'
# data_dir = '/home/sysadmin/Ashish/LatestGAN_Fake_ClassificationModel/6000_TrainData'
# data_dir = '/home/sysadmin/Ashish/LatestGAN_Fake_ClassificationModel/latestGAN_v4.0_10000images_UpdatedGlaucoma_images'
# data_dir = '/home/sysadmin/Ashish/latestGAN_v3.0_100000'

#validation from scratch
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/40000FakeImages_training'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/40000FakeImages_training/Models/PytorchMethod1(Generic)/'

# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/40000FakeImages+realNormalImages_training'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/40000FakeImages+realNormalImages_training/Models/PytorchMethod1(Generic)/'

# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/50000FakeImages+realImages_traiining'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/50000FakeImages+realImages_traiining/Models/PytorchMethod1(Generic)'

# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/RealImages(1590,523+627)_Training'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/RealImages(1590,523+627)_Training/Models/PytorchMethod1(Generic)/'

# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/50000Fake+900RealImages_Training'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/50000Fake+900RealImages_Training/Models/PytorchMethod1(Generic)/'

# 50000 fake images only (from latest GAN method for Nov4th meeting : without test images)-- no real images included for classification
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/50000_FakeImagesOnly_training'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/50000_FakeImagesOnly_training/Models/PytorchMethod1(Generic)'

# 40000 fake images only (gan used for Nov4th meeting - )
# data_dir = '/home/sysadmin/Ashish/40000_Classification_OnlyFake'
# modelSavingPath = '/home/sysadmin/Ashish/40000_Classification_OnlyFake/Models/PytorchMethod1(Generic)/'

# Finalized GAN - Real Images - 1100 training + 100 valid + 72 test : RNFL based data split for Glaucoma
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA/Models/PytorchMethod1(Generic)/'

# Finalized GAN - Fake Images - 50000G + 50000 N ( from 1100 G + 1100 N training: RNFL based data split for Glaucoma)
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/FinalizedGAN_Severity_GANImages'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/FinalizedGAN_Severity_GANImages/Models/GenericModel/'

# Finalized GAN - Fake Images - 50000G + 50000 N ( from 1000 G + 1000 N training, 10000 N + 5000G - valid: Severity based data split for Glaucoma)
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/FInalized_GAN_basedOnSeverity_FakeOnly'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/FInalized_GAN_basedOnSeverity_FakeOnly/Models/GenericModel/'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/FInalized_GAN_basedOnSeverity_FakeOnly/Models/scratch_model/10Epoch/' #scratch model - VGG 10 Epoch
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/FInalized_GAN_basedOnSeverity_FakeOnly/Models/scratch_model/20Epoch/' #scratch model - VGG 20 Epoch
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/FInalized_GAN_basedOnSeverity_FakeOnly/Models/scratch_model/resnet/10Epoch/' #  scratch model - resnet 10 epoch

# Finalized GAN - Fake Images - (five thousand) 5000G + 5000 N ( from 1000 G + 1000 N training, 1000 N + 1000G - valid: Severity based data split for Glaucoma)
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/Models/GenericModel/'

# Finalized GAN - Fake Images - (five thousand) 5000G + 5000 N ( from 1000 G + 1000 N training, 1000 N + 1000G - valid: Severity based data split for Glaucoma)
# for grayscale - (CAM visualization)
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/Models/CAM/'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/Models/Check/feature_extractor/' # feature extractor model
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/Models/Check/feature_extractor_20Epoch' # feature extractor model
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/Models/Check/scratch_model_30Epoch/' # scratch - full training model 30 epoch
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/Models/Check/scratch_model/' # scratch - full training

# Finalized GAN - Real Images - 1000 training + 100 valid + 78 test : Severity based data split for Glaucoma
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/Models/PytorchMethod1(Generic)/'

# Finalized GAN - ******* FLATTENED Real Images ******* - 1000 training + 100 valid + 78 test : Severity based data split for Glaucoma
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_Severity_FlattemedImages/'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_Severity_FlattemedImages/Models/PytorchMethod1(Generic)/'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_Severity_FlattemedImages/Models/scratch_model/'

# Finalized GAN - ******* FLATTENED Fake Images ******* - 5000 training + 1000 valid + 100 test : Severity based data split for Glaucoma
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/Flattened_GAN_FakeImages_Results/'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/Flattened_GAN_FakeImages_Results/Models/PytorchMethod1(Generic)/'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/Flattened_GAN_FakeImages_Results/Models/scratch_model/' # scratch model

# MODEL COMBINATIONS
#Model1
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model1_RNRG'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model1_RNRG/Models/feature_extracted/'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model1_RNRG/Models/resnet_finetuned/'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model1_RNRG/Models/scratch_model/VGG/' # vgg - scratch model : 10 Epochs
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model1_RNRG/Models/scratch_model/VGG/20Epoch/' # vgg - scratch model : 20 Epochs
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model1_RNRG/Models/scratch_model/VGG/40Epoch/'  # vgg - scratch model : 40 Epochs
#Model2
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model2_RNFG'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model2_RNFG/Models/CAM_resnet/'
#Model3
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model3_FNRG'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model3_FNRG/Models/CAM_resnet/'
#Model4 - 1000 N + 1000 G
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/resnet_finetuned/' # ResNet finetuned
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/resnet_ft_scratch/'
#VGG
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/VGG_finetuned/' # VGG
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/vgg_finetune_v2_pretrained/' # VGG - 2nd try
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/vgg_finetune_v2_scratch/' # VGG - 3rd try
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/vgg_finetune_v2_scratch_20Epoch/' # VGG - 4th try
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/vgg_finetune_v2_scratch_40Epoch/' # VGG - 5th try 40 Epoch (scratch model)
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/scratch_models/VGG/60Epoch/'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/scratch_models/VGG/100Epochs/'

#Model4 - 5000N + 5000G
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_FNFG/'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_FNFG/Models/resnet_finetuned/'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_FNFG/vgg_scratch_50Epoch/'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_FNFG/test_CAM_Scratch/VGG/10Epoch/'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_FNFG/test_CAM_Scratch/VGG/20Epoch/'
# --------------------------------------------------------------------------------------------------------------

# UNET images
# real images
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/UNet_Segmentation_RealImages/classif_model/'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/UNet_Segmentation_RealImages/classif_model/Models/Resnet_finetuned/'

#fake images - "1000 N + 1000 G" (from the 5000 N, 5000G) fake images generated using GAN
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/UNet_Segmentation/classif_model/'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/UNet_Segmentation/classif_model/Models/Resnet_finetuned/'


# *********** GANInput_RealData_Filtered ************
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/GANInput_RealData_Filtered/'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/GANInput_RealData_Filtered/Models/scratch_model/20Epoch/' #20 epoch
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/GANInput_RealData_Filtered/Models/scratch_model/30Epoch/' #30 epoch
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/GANInput_RealData_Filtered/Models/scratch_model/50Epoch/' #50 epoch
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/GANInput_RealData_Filtered/Models/scratch_model/60Epoch/'

# ************ GANINput_SyntheticData_Filtered *********
data_dir = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered/'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered/Models/scratch_model/20Epoch/'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered/Models/scratch_model/30Epoch/'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered/Models/scratch_model/50Epoch/'
modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered/Models/scratch_model/50Epoch_v2/' # trial 2 for 50 epoch
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered/Models/scratch_model/60Epoch/'



# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
# model_name = "resnet"
model_name = "vgg"

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for
# num_epochs = 10
# num_epochs = 20
# num_epochs = 30
# num_epochs = 40
num_epochs = 50
# num_epochs = 60
# num_epochs = 100

# Flag for feature extracting. When False, we fine tune the whole model,
# When True we only update the reshaped layer params
feature_extract = False # usual scenario
# feature_extract = True # for validating CAM for GAN fake images only classification

# device = torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

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

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('best model epoch', epoch)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)

        if (epoch%2 == 0):
            # progressPath = 'Latest_GAN_Resnet_'+str(epoch)+'.pt'
            # progressPath = model_name+'_originalImages_'+str(epoch)+'.pt'
            # progressPath = model_name+'_6000(v2)_FakeImages_Epoch:'+str(epoch)+'.pt'
            # progressPath = model_name + '_selectedImages_GAN_' + str(epoch) + '.pt'
            progressPath = modelSavingPath+model_name+'_'+str(dataset_sizes['train'])+'_'+str(epoch)+'.pt'
            # torch.save(model.state_dict(), progressPath)
            torch.save(model, progressPath)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

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

# Print the model we just instantiated
print(model_ft)

# for coloured images
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.Resize((input_size,input_size)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'valid': transforms.Compose([
#         transforms.Resize((input_size,input_size)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'test': transforms.Compose([
#         transforms.Resize((input_size,input_size)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }

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

print("Initializing Datasets and Dataloaders...")

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid', 'test']}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'valid', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
print(dataset_sizes)
print('Training image count:',dataset_sizes['train'])
class_names = image_datasets['train'].classes

# Send the model to GPU
model_ft = model_ft.to(device)

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
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
# optimizer_ft = optim.SGD(params_to_update, lr=0.0001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

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

def evaluate_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    class_names = ('glaucoma', 'normal')
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    originalLabels = []
    predictedLabels = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders_dict['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true += list(labels.cpu().numpy())
            y_pred += list(preds.cpu().numpy())
            total += labels.size(0)
            correct += (preds == labels.data).sum().item()

            for j in range(inputs.size()[0]):
                # images_so_far += 1
                # ax = plt.subplot(num_images // 2, 2, images_so_far)
                # ax.axis('off')
                # ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                # imshow(inputs.cpu().data[j])
                #
                # if images_so_far == num_images:
                #     model.train(mode=was_training)
                #     return

                originalLabels.append(class_names[labels[j]])
                predictedLabels.append(class_names[preds[j]])

        print('Test accuracy using Pytorch official doc method : {:0.2f} %'.format(100 * (correct / total)))
        model.train(mode=was_training)

        # evaluation metrics
        print(metrics.accuracy_score(y_true, y_pred))
        from sklearn.metrics import classification_report
        print(classification_report(y_true, y_pred, target_names=['glaucoma', 'normal']))
        # print(classification_report(originalLabels,predictedLabels,labels=labels))

        from sklearn.metrics import confusion_matrix
        labels = ['glaucoma', 'normal']
        # print(confusion_matrix(y_true,y_pred,labels))
        print(confusion_matrix(originalLabels, predictedLabels, labels=labels))
        from sklearn.metrics import roc_auc_score
        print(roc_auc_score(y_true, y_pred))

# Save
# savedModelName = 'Consolidated_Resnet_Model_RealImages.pt'
# savedModelName = 'Consolidated_Resnet_Model_FakeImages.pt'
# savedModelName = 'Consolidated_Resnet_Model_FakeImages_20000.pt' # have to train again
# savedModelName = 'latestGAN_40000.pt'
# savedModelName = model_name+'_originalImages.pt'
# savedModelName = model_name+'_6000(v2)_FakeImages_Epochs'+str(num_epochs)+'.pt'
# savedModelName = model_name+'1lkhTrainImages'+'_selectedImages_GAN_pt'
savedModelName = modelSavingPath+model_name+'_'+str(dataset_sizes['train'])+'_bestEpoch'+'.pt'

torch.save(model_ft, savedModelName)

#evaluate model on test images
evaluate_model(model_ft)
print('model is : ', savedModelName)
