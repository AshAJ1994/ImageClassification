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
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/FinalizedGAN_Severity_GANImages/Models/GenericModel'

# Finalized GAN - Real + Fake Images - 50000G ,1100 G + 50000 N, 1100 N ( from 1100 G + 1100 N training: RNFL based data split for Glaucoma)
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/FinalizedGAN_Severity_GANImages+RealImages(1100)'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/FinalizedGAN_Severity_GANImages+RealImages(1100)/Models/GenericModel'

# Finalized GAN - Fake Images - 50000G,1000G + 50000 N, 1000 N ( from 1000 G + 1000 N training, 10000 N + 5000G - valid: Severity based data split for Glaucoma)
# data_dir = '/home/sysadmin/Ashish_PGAN_Validation/FInalized_GAN_basedOnSeverity_Fake+Real(1000)'
# modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/FInalized_GAN_basedOnSeverity_Fake+Real(1000)/Models/GenericModel/'

# Finalized GAN - Fake Images(5000) + Real Images - (five thousand) 5000G + 5000 N ( from 1000 G + 1000 N training, 10000 N + 5000G - valid: Severity based data split for Glaucoma)
data_dir = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000Fake+1000Real'
modelSavingPath = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000Fake+1000Real/Models/GenericModel/'

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for
num_epochs = 10
# num_epochs = 20

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

device = torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else "cpu")


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
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
print(model_ft)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
