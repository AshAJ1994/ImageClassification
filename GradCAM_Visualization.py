import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
# calculate accuracy
from sklearn import metrics
from collections import Counter
import cv2
from torch import topk
from torch.nn import functional as F
import skimage.transform
from os.path import splitext
from pathlib import Path
from PIL import Image
import itertools
import os

# CAM visualization trial - Ants Bees dataset
# PATH = '/home/sysadmin/Ashish/ants_bees_dataset/Models/Resnet/Resnet_224_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish/ants_bees_dataset/Models/BW_Resnet/Resnet_224_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish/ants_bees_dataset/Models/BW_Resnet/Resnet_224_bestEpoch.pt'

# for CAM visualization - trained on grayscale images
# real images - test path containing 78 images per class : RNFL based classification - (1000 G + 1000 N training)
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/Models/CAM/B&W/Resnet_2000_bestEpoch.pt'

#for CAM visualization - trained on FLATTENED Real Images - Grayscale (1000N + 1000G train , 78N + 78G test)
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_Severity_FlattemedImages/Models/CAM/B&W/Resnet_2000_bestEpoch.pt'

#for CAM visualization - trained on GAN fake images - 5000N + 5000G
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/Models/GenericModel/resnet_10000_bestEpoch.pt' # fine tuned
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/Models/Check/feature_extractor/resnet_10000_bestEpoch.pt' # feature extractor
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/Models/Check/feature_extractor_20Epoch/feature_extractor_20Epochresnet_10000_bestEpoch.pt' # feature extractor - 20 epochs
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/Models/Check/scratch_model/resnet_10000_bestEpoch.pt' # scratch model
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/Models/Check/finetuned_model_20Epoch/resnet_10000_26.pt' # scratch model - 30 epochs

#Model Combinations - CAM visualization
#model1
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model1_RNRG/Models/feature_extracted/resnet_2000_bestEpoch.pt' # feature extracted
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model1_RNRG/Models/resnet_finetuned/resnet_2000_bestEpoch.pt' # fine tuned
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/Models/CAM/Resnet_2000_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model1_RNRG/Models/scratch_model/VGG/10Epoch/vgg_2000_bestEpoch.pt' # vgg scratch model - 10 Epoch
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model1_RNRG/Models/scratch_model/VGG/20Epoch/vgg_2000_bestEpoch.pt'  # vgg scratch model - 20 Epoch
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model1_RNRG/Models/scratch_model/VGG/40Epoch/vgg_2000_bestEpoch.pt' # vgg scratch model - 40 Epoch
# refer to above real images model
#model2
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model2_RNFG/Models/CAM_resnet/resnet_2000_bestEpoch.pt'
#Model3
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model3_FNRG/Models/CAM_resnet/resnet_2000_bestEpoch.pt'
#Model 4 - 1000 N + 1000 G
# vgg model trained on fake images only
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/VGG_finetuned/vgg_2000_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/vgg_finetune_v2/vgg_2000_bestEpoch.pt' # VGG - 2nd try
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/vgg_finetune_v2_scratch/vgg_2000_bestEpoch.pt' # vgg - 3rd try 10 Epochs (scratch model)
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/vgg_finetune_v2_scratch_20Epoch/vgg_2000_bestEpoch.pt' # vgg - 4th try : 20 Epochs (scratch model)
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/vgg_finetune_v2_scratch_40Epoch/vgg_2000_bestEpoch.pt' # vgg - 40 Epoch - 5th try scratch model
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/scratch_models/VGG/60Epoch/vgg_2000_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/scratch_models/VGG/60Epoch/vgg_2000_50.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/scratch_models/VGG/100Epochs/vgg_2000_40.pt'

# resnet model
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/resnet_finetuned/resnet_2000_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/resnet_ft_scratch/resnet_2000_bestEpoch.pt' # scratch model

# Model4 - 5000N + 5000G
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_FNFG/Models/resnet_finetuned/resnet_10000_8.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_FNFG/vgg_scratch_50Epoch/vgg_10000_36.pt' # vgg scratch 50 epoch fake images model
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_FNFG/test_CAM_Scratch/VGG/10Epoch/vgg_10000_bestEpoch.pt' # vgg scratch 10 epoch fake images model
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_FNFG/test_CAM_Scratch/VGG/10Epoch/vgg_10000_6.pt' # not the best epoch , but gave better classification results
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_FNFG/test_CAM_Scratch/VGG/20Epoch/vgg_10000_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_FNFG/test_CAM_Scratch/VGG/20Epoch/vgg_10000_4.pt'

# GANInput_SyntheticData_Filtered - Fake images - Train 600 N + 600 G
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered/Models/scratch_model/20Epoch/vgg_1200_bestEpoch.pt' # 20 Epoch scratch model
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered/Models/scratch_model/30Epoch/vgg_1200_bestEpoch.pt' # 30 epoch scratch model
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered/Models/scratch_model/50Epoch/vgg_1200_48.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered/Models/scratch_model/50Epoch/vgg_1200_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered/Models/scratch_model/50Epoch_v2/vgg_1200_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered/Models/scratch_model/50Epoch_v2/vgg_1200_30.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered/Models/scratch_model/60Epoch/vgg_1200_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered/Models/scratch_model/60Epoch/vgg_1200_48.pt'

#version 2 - after filtering GAN input images from real data - 862/1000 for G , 990/1000 for N
# GANInput_SyntheticData_Filtered - V2  - Fake images - Train 600 N + 600 G ,
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered_v2_600/Models/scratch_models/vgg/10Epoch/vgg_1200_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered_v2_600/Models/scratch_models/vgg/50Epoch/vgg_1200_30.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered_v2_600/Models/scratch_models/vgg/50Epoch/vgg_1200_48.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered_v2_600/Models/scratch_models/vgg/50Epoch/vgg_1200_10.pt'
# PATH_A = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered_v2_600/Models/scratch_models/vgg/50Epoch/vgg_1200_48.pt'
# PATH_B = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered_v2_600/Models/scratch_models/vgg/50Epoch/vgg_1200_30.pt'

#version 3 - after filtering GAN input images from real data - 862/1000 for G , 990/1000 for N
# GANInput_SyntheticData_Filtered - V3  - Fake images - Train 2000 N + 2000 G ,
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered_v3_2000/Models/scratch_models/vgg/50Epoch/vgg_4000_20.pt'

#version 4 - after filtering GAN input images from real data - 862/1000 for G , 990/1000 for N
# GANInput_SyntheticData_Filtered - V4  - Fake images - Train 5000 N + 5000 G ,
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered_v4_5000/Models/scratch_models/vgg/50Epoch/vgg_10000_6.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered_v4_5000/Models/scratch_models/vgg/50Epoch/vgg_10000_20.pt'
# PATH_B = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered_v4_5000/Models/scratch_models/vgg/50Epoch/vgg_10000_6.pt'

# GANInput_RealData_Filtered - Real Images - Train : 600N + 600G
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GANInput_RealData_Filtered/Models/scratch_model/20Epoch/vgg_1200_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GANInput_RealData_Filtered/Models/scratch_model/30Epoch/vgg_1200_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GANInput_RealData_Filtered/Models/scratch_model/50Epoch/vgg_1200_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GANInput_RealData_Filtered/Models/scratch_model/60Epoch/vgg_1200_bestEpoch.pt'
# PATH_A = '/home/sysadmin/Ashish_PGAN_Validation/GANInput_RealData_Filtered/Models/scratch_model/50Epoch/vgg_1200_bestEpoch.pt'

# PATH = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_30000_Images/50Epoch/earlyStopping_checkpoint.pt'
# PATH = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_5000_Images/50Epoch/earlyStopping_checkpoint.pt'
# PATH = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_5000_Images/60Epoch_patience10/earlyStopping_checkpoint.pt'
# PATH_B = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_5000_Images/50Epoch/earlyStopping_checkpoint.pt'
# PATH_A = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_600_Images/60Epoch_patience10/earlyStopping_checkpoint.pt'

#early stopping - fake model - validated on fake images
# PATH = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_600_Images/60Epoch_patience5/Fake_Valid/earlyStopping_checkpoint.pt'
PATH_A = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_600_Images/60Epoch_patience5/Fake_Valid/earlyStopping_checkpoint.pt'
# PATH = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_5000_Images/60Epoch_Patience5/Fake_Valid/earlyStopping_checkpoint.pt'
PATH_B = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_5000_Images/60Epoch_Patience5/Fake_Valid/earlyStopping_checkpoint.pt'

# PATH = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Real_600_Images/60Epoch/earlyStopping_checkpoint.pt'
# PATH = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Real_600_Images/60Epoch_Patience10/earlyStopping_checkpoint.pt'
# PATH_A = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Real_600_Images/60Epoch_Patience10/earlyStopping_checkpoint.pt'

# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GANInput_RealData_Filtered/Models/dummy_models/vgg_1200_bestEpoch.pt'
PATH = '/home/sysadmin/PycharmProjects/ImageClassification/EarlyStopping_Results/Fake_30000_Images/60Epoch_patience5/Fake_Valid/fake_es_cp.pt'

#Unet images - CAM classification
# real images
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/UNet_Segmentation_RealImages/classif_model/Models/Resnet_finetuned/resnet_2000_bestEpoch.pt'

# fake images -  test : 500N + 500G, train: 5000N + 5000G, valid: 1000N + 1000G
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/UNet_Segmentation/classif_model/Models/Resnet_finetuned/resnet_2000_bestEpoch.pt'

# fake images - train: 50000 N + 50000 G
# scratch model
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FInalized_GAN_basedOnSeverity_FakeOnly/Models/scratch_model/10Epoch/vgg_100000_bestEpoch.pt' # vggg scratch 10 Epoch model
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FInalized_GAN_basedOnSeverity_FakeOnly/Models/scratch_model/20Epoch/vgg_100000_bestEpoch.pt' # vgg scratch 20 epoch model
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FInalized_GAN_basedOnSeverity_FakeOnly/Models/scratch_model/resnet/10Epoch/resnet_100000_bestEpoch.pt' # resnet scratch 10 Epoch model

#flattened fake images - train : 5000 N + 5000 G
#scratch VGG model
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Flattened_GAN_FakeImages_Results/Models/scratch_model/vgg_10000_bestEpoch.pt'


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

# ants bees dataset - test data path
# test_path = '/home/sysadmin/Ashish/ants_bees_dataset/test/'

# real images - test path containing 78 images per class : RNFL based classification - (1000 G + 1000 N training)
test_path = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/test'

#fake images - 78G + 78N
# test_path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/test/'
# renamed Fake test images - 78N + 78G
# test_path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/test_renamed/'

#fake images - test : 2000N + 2000G , train - 50000N + 50000G
# test_path = '/home/sysadmin/Ashish_PGAN_Validation/FInalized_GAN_basedOnSeverity_FakeOnly/test/'

# FLATTENED real images - test path containing FLATTENED 78 images per class : RNFL based classification - (1000 G + 1000 N training)
# test_path = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_Severity_FlattemedImages/test/'

# gan images - (trained using 5000N + 5000G train data )
# test_path = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/test/'

#UNet images
#real images
# test_path = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/UNet_Segmentation_RealImages/classif_model/test/'

# UNet fake images - "78 G + 78 N" (picked from test : 500N + 500G, train: 5000N + 5000G, valid: 1000N + 1000G)
# test_path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/UNet_Segmentation/classif_model/test/'

class CustomImageFolder(datasets.ImageFolder):

    def __getitem__(self, index):
        original_tuple = super(CustomImageFolder, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple+(path, ))
        # return path
        return tuple_with_path

# test_data = datasets.ImageFolder(root=test_path, transform=transforms.Compose([
#                                       transforms.Grayscale(num_output_channels=3),
#                                       transforms.Resize((224, 224)),
#                                       # transforms.ToPILImage(),
#                                       transforms.ToTensor(),
#                                       # transforms.Normalize((0.5,), (0.5,),(0.5,)),
#                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
# ]))
test_data = CustomImageFolder(root=test_path, transform=transforms.Compose([
                                      transforms.Grayscale(num_output_channels=3),
                                      transforms.Resize((224, 224)),
                                      # transforms.ToPILImage(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,),(0.5,)),
                                      # transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
]))
print(len(test_data))
test_data_loader = DataLoader(test_data, shuffle=False, batch_size=1, num_workers=0)
print(dict(Counter(test_data_loader.dataset.targets)))

display_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((188,256))
])
display_transform_flattenedImage = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((94,256))
])

imgConversion_transform = transforms.Compose([
    transforms.ToPILImage()
])

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:0,1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else "cpu")

# Load
model = torch.load(PATH)
model = model.to(device)

model_A = torch.load(PATH_A)
model_A = model_A.to(device)

model_B = torch.load(PATH_B)
model_B = model_B.to(device)

# model = model_ft.load_state_dict(PATH)
model.eval()
was_training = model.training

model_A.eval()
was_training = model_A.training
model_B.eval()
was_training = model_B.training

class_names = ('glaucoma', 'normal')
# class_names = ('ants', 'bees')

correct = 0
total = 0
y_true = []
y_pred = []
y_pred_prob = []

y_true_A = []
y_pred_A = []
y_pred_prob_A = []

y_true_B = []
y_pred_B = []
y_pred_prob_B = []

originalLabels = []
predictedLabels = []

#Evaluation of Model - Confusion Matrix Plot
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

import torch
import torchvision
from torchcam.cams import CAM, GradCAM, SmoothGradCAMpp
# from pytorch_grad_cam import GrfadCam # pypi grad cam library

class CAM:
    def __init__(self, model, target_layer, use_cuda=False):
        self.model = model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.activations_and_grads = ActivationsAndGradients(self.model, target_layer)

    def forward(self, input_img):
        return self.model(input_img)

    def gradcampp(self, activations, grads):
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(1, 2))
        eps = 0.00000001
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations[:, None, None] * grads_power_3 + eps)

        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=(1, 2))
        return weights

    def scorecam(self,
                 input_tensor,
                 activations,
                 target_category,
                 original_score):
        with torch.no_grad():
            upsample = torch.nn.UpsamplingBilinear2d(size=input_tensor.shape[2:])
            activation_tensor = torch.from_numpy(activations).unsqueeze(0)
            if self.cuda:
                activation_tensor = activation_tensor.cuda()

            upsampled = upsample(activation_tensor)
            upsampled = upsampled[0,]

            maxs = upsampled.view(upsampled.size(0), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0), -1).min(dim=-1)[0]
            maxs, mins = maxs[:, None, None], mins[:, None, None]
            upsampled = (upsampled - mins) / (maxs - mins)

            input_tensors = input_tensor * upsampled[:, None, :, :]
            batch_size = 16
            scores = []
            for i in range(0, input_tensors.size(0), batch_size):
                batch = input_tensors[i: i + batch_size, :]
                outputs = self.model(batch).cpu().numpy()[:, target_category]
                scores.append(outputs)
            scores = torch.from_numpy(np.concatenate(scores))
            weights = torch.nn.Softmax(dim=-1)(scores - original_score).numpy()
            return weights

    def __call__(self, input_tensor, method="gradcam", target_category=None):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_grads(input_tensor)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()[0, :]
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()[0, :]
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        if method == "gradcam++":
            weights = self.gradcampp(activations, grads)
        elif method == "gradcam":
            weights = np.mean(grads, axis=(1, 2))
        elif method == "scorecam":
            original_score = original_score = output[0, target_category].cpu()
            weights = self.scorecam(input_tensor,
                                    activations,
                                    target_category,
                                    original_score=original_score)
        else:
            # raise "Method not supported"
            print('Method not supported')

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_tensor.shape[2:][::-1])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        if np.isnan(cam).any():
            print('image problem - nan values found ')
        cam = np.uint8(255 * cam) # line added by me (Ashish)
        return cam

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = []
        self.activations = []

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        self.gradients = [grad_input[0]] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

from torchvision.transforms import Compose, Normalize, ToTensor

def preprocess_image(img: np.ndarray, mean=None, std=None) -> torch.Tensor:
    if std is None:
        std = [0.5, 0.5, 0.5]
    if mean is None:
        mean = [0.5, 0.5, 0.5]

    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])

    return preprocessing(img.copy()).unsqueeze(0)

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)

def show_cam_on_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_resized = cv2.resize(mask, (256,188))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def saveCAMImage(img, cam, resultText, saveDirPath):
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(cam, (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.5 + img * 0.5

    combine = np.concatenate([img, result], 1)
    p = Path(filenames[0])
    saveName = saveDirPath + p.name
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (10, 10)
    # org = (10, 80)
    # fontScale
    fontScale = 0.4
    # Blue color in BGR
    color = (255, 255, 255)
    # Line thickness of 2 px
    thickness = 1
    cv2.putText(combine, resultText, org=org, color=color, fontFace=font, fontScale=fontScale, thickness=thickness)
    cv2.imwrite(saveName, combine)

predClassVsRNFLDict = {}

# with torch.no_grad():
for images,labels,filenames in test_data_loader:

    images = images.to(device)
    labels = labels.to(device)

    images.requires_grad = True

    prediction = model(images)
    # pred_probabilities = F.softmax(prediction).data.cpu().squeeze() # deprecated method
    pred_probabilities = F.softmax(prediction, dim=1)
    pred_probabilities_sigmoid = F.sigmoid(prediction)
    class_idx = topk(pred_probabilities, 1)[1].int()
    l = str(float("{:.2f}".format(torch.max(pred_probabilities))))
    # val = float(torch.max(pred_probabilities_sigmoid))
    # val = float(pred_probabilities[:,1])
    val = float(pred_probabilities_sigmoid[:, 1])
    buf = 'predicted class :' + class_names[class_idx] + '   probability : ' + l + '  actual class : ' + class_names[
        int(labels[0])]

    target_layer = model.features[-1] # for VGG
    # target_layer = model.layer4[-1] # for Resnet
    method = "gradcam"  # Can be gradcam/gradcam++/scorecam

    # cam = CAM(model=model, target_layer=target_layer, use_cuda=True)
    # grayscale_cam = cam(input_tensor=images, target_category=1)
    #
    # img = cv2.imread(filename=filenames[0])
    # # visualization = show_cam_on_image(img, grayscale_cam)
    # if np.isnan(grayscale_cam).any():
    #     print('Image file has a problem : ', filenames[0])

    # path = '/home/sysadmin/Ashish/ants_bees_dataset/test_CAM_hookBased/' # ants vs bees
    # path = '/home/sysadmin/Ashish/ants_bees_dataset/test_CAM_hookbased_v2/' # ants vs bess - repeated exp to check the settings for CAM
    # path = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/test_CAM_hookBased/' # real normal vs glaucoma
    # path = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_Severity_FlattemedImages/test_CAM_hookBased/' # flattened real normal vs glaucoma
    # path = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/test_CAM_hookBased(FakeModelOnFake)/' # fake images N vs G
    # path = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/test_CAM_hookBased(RealModelOnFake)/'
    # path = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/test_CAM_hookBased(FakeModelOnReal)/' # fake model on real images
    # path = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/test_CAM_hookBased(FakeModelOnReal)_v2/' # fake model on real images(updated version)
    # path = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/test_FullTraining_CAM_hookBased(FakeModelOnReal)/'
    # path = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/test_CAM_FonR_featureextractor /'
    # path = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/test_CAM_FonR_featureextractor_20epoch/' # fake model (20 epochs) on real test images -
    # path = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/test_CAM_scratch_FonR/'
    # path = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/test_CAM_scratch_FonF/'

    #Model combinations
    #model1
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model1_RNRG/test_CAM/'
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model1_RNRG/test_CAM_featureExtracted/'
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model1_RNRG/Models/test_CAM_finetuned/'
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model1_RNRG/CAM_RonR/'
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model1_RNRG/CAM_RonF/'
    #Model2
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model2_RNFG/test_CAM/'
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model2_RNFG/CAM_ModelonR/'
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model2_RNFG/CAM_ModelonF/'
    #Model3
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model3_FNRG/test_CAM/'
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model3_FNRG/CAM_ModelonR/'
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model3_FNRG/CAM_ModelonF/'
    #Model4
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_FNFG/test_CAM_finetuned/'
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_FNFG/test_CAM_featureextracted/'
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/CAM_FonR/'
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/CAM_FonF/'
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/test_CAM_scratch/vgg/FonR/' # vgg 10 Epoch fake model on real images
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/test_CAM_scratch/vgg/FonF/' # vgg 10 Epoch fake model on fake images
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/test_CAM_scratch/vgg/20Epoch/FonR/' #vgg 20 Epoch fake model on real images
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/test_CAM_scratch/vgg/40Epoch/FonR/' #vgg 40 epoch fake model on real images
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/test_CAM_scratch/vgg/40Epoch/FonF/' #vgg 40 epoch fake model on fake images
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/test_CAM_scratch/vgg/40Epoch_v2/Glaucoma_TargetLayer/FonR/' #vgg 40 epoch fake model on fake images - glaucoma target layer

    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/test_CAM_scratch/vgg/40Epoch_v3_gradcam++/FonR/'
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/test_CAM_scratch/resnet/FonR/' # resnet fake model on real images

    # Train - 5000 N + 5000 G - fake images
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_FNFG/test_CAM_Scratch/VGG/50Epoch/FonR/' #vgg scratch fake images model on real images
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_FNFG/test_CAM_Scratch/VGG/10Epoch/FonR/' # 10 epochs scratch
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_FNFG/test_CAM_Scratch/VGG/20Epoch/FonR/' # 20 epochs scratch
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_FNFG/test_CAM_Scratch/VGG/20Epoch/FonR_bestEpoch/'

    # Train - 50000 N + 50000 G - fake images
    # path = '/home/sysadmin/Ashish_PGAN_Validation/FInalized_GAN_basedOnSeverity_FakeOnly/test_CAM/10Epoch/FonR/' # 10 Epoch vgg scratch model - fake model on real test images
    # path = '/home/sysadmin/Ashish_PGAN_Validation/FInalized_GAN_basedOnSeverity_FakeOnly/test_CAM/10Epoch/FonF/' # 10 Epoch VGG scratch model - fake model on fake images
    # path = '/home/sysadmin/Ashish_PGAN_Validation/FInalized_GAN_basedOnSeverity_FakeOnly/test_CAM/20Epoch/FonR/' # 20 Epoch vgg scratch model - modelfake model on real test images

    #GANFiltered-RealInputImages - 600N + 600G,
    # path = '/home/sysadmin/Ashish_PGAN_Validation/GANInput_RealData_Filtered/test_CAM_scratch/20Epoch/FonR/' #20Epoch
    # path = '/home/sysadmin/Ashish_PGAN_Validation/GANInput_RealData_Filtered/test_CAM_scratch/30Epoch/FonR/' #30Epoch

    # GANFiltered-FakeInputImages - 600N + 600G
    # path = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered/test_CAM_scratch/20Epoch/FonR/' # 20 EPoch
    # path = '/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered/test_CAM_scratch/30Epoch/FonR/' # 30 EPoch


    # saveCAMImage(img, grayscale_cam, buf, saveDirPath=path)

    #UNET images
    # real images
    # path = test_path + 'test_CAM/'
    # fake images
    # path = test_path + 'test_CAM_FonF/'
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/UNet_Segmentation/classif_model/test/test_CAM_FonR/'
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/test_CAM_vgg_ft/' # VGG - 1000n + 1000g
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/test_CAM_resnet_ft/' # resnet 1000 n + 1000g
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_FNFG/Models/test_CAM_resnetresults/' # resnet 5000 N + 5000 G

    # using pytorch official doc method for calculating loss / correct predictions
    outputs = model(images)
    pred_probabilities_sigmoid = F.sigmoid(outputs)
    val = float(pred_probabilities_sigmoid[:, 1])
    prob, preds = torch.max(outputs, 1) #doubtful
    # print('preds',preds)
    y_true += list(labels.cpu().numpy())
    y_pred += list(preds.cpu().numpy())
    # y_pred_prob += list(prob.cpu().detach().numpy())
    y_pred_prob.append(val)

    total += labels.size(0)
    correct += (preds == labels.data).sum().item()

    outputs_A = model_A(images)
    pred_probabilities_sigmoid_A = F.sigmoid(outputs_A)
    val_A = float(pred_probabilities_sigmoid_A[:, 1])
    prob_A, preds_A = torch.max(outputs_A, 1)
    # print('preds',preds)
    y_true_A += list(labels.cpu().numpy())
    y_pred_A += list(preds.cpu().numpy())
    # y_pred_prob_A += list(prob.cpu().detach().numpy())
    y_pred_prob_A.append(val_A)

    outputs_B = model_B(images)
    pred_probabilities_sigmoid_B = F.sigmoid(outputs_B)
    val_B = float(pred_probabilities_sigmoid_B[:, 1])
    prob_B, preds_B = torch.max(outputs_B, 1)
    # print('preds',preds)
    y_true_B += list(labels.cpu().numpy())
    y_pred_B += list(preds.cpu().numpy())
    # y_pred_prob_B += list(prob.cpu().detach().numpy())
    y_pred_prob_B.append(val_B)


    baseName = os.path.basename(filenames[0])
    predClassVsRNFLDict[baseName] = int(preds)

    for j in range(images.size()[0]):
        originalLabels.append(class_names[labels[j]])
        predictedLabels.append(class_names[preds[j]])

import pandas as pd
predClassVsRNFLDF = pd.DataFrame(list(predClassVsRNFLDict.items()), columns=['Image_Filename','Predicted_Class'])
realTestImages_unetThickness_DF = pd.read_csv('RealTestImages_UnetBasedThickness.csv',sep=',', index_col=0)
# fakeTestImages_unetThickness_DF = pd.read_csv('FakeTestImages_UnetBasedThickness.csv', sep=',', index_col=0)
# fakeTestImages_unetThickness_DF = pd.read_csv('Renamed_FakeTestImages_UnetBasedThickness.csv', sep=',', index_col=0)
realImages_mergedDF = pd.merge(predClassVsRNFLDF, realTestImages_unetThickness_DF, on='Image_Filename', how='left')
# fakeImages_mergedDF = pd.merge(predClassVsRNFLDF, fakeTestImages_unetThickness_DF, how='left', on='Image_Filename')

print('Test accuracy using Pytorch official doc method : {:0.2f} %'.format(100 * (correct / total)))
from sklearn.metrics import roc_curve,roc_auc_score,auc
# real images
fpr, tpr, thrsh = metrics.roc_curve(realImages_mergedDF['Predicted_Class'], realImages_mergedDF['Avg_RNFLThickness'], pos_label=1)
auc_score_predictedClass = auc(fpr, tpr)
fpr, tpr, thrsh = metrics.roc_curve(realImages_mergedDF['Actual_Class'], realImages_mergedDF['Avg_RNFLThickness'], pos_label=1)
auc_score_RealClass = auc(fpr, tpr)
#fake images
# fpr, tpr, thrsh = metrics.roc_curve(fakeImages_mergedDF['Predicted_Class'], fakeImages_mergedDF['Avg_RNFLThickness'], pos_label=1)
# auc_score_predictedClass = auc(fpr, tpr)
# fpr, tpr, thrsh = metrics.roc_curve(fakeImages_mergedDF['Actual_Class'], fakeImages_mergedDF['Avg_RNFLThickness'], pos_label=1)
# auc_score_RealClass = auc(fpr, tpr)

# evaluation metrics
# print(metrics.accuracy_score(y_true, y_pred))
print('AUC score from predict probabilities', metrics.roc_auc_score(y_true, y_pred_prob))
# print(metrics.accuracy_score(realImages_mergedDF['Actual_Class'], realImages_mergedDF['Predicted_Class']))
# print(metrics.accuracy_score(fakeImages_mergedDF['Actual_Class'], fakeImages_mergedDF['Predicted_Class']))

print('AUC score from predict probabilities - Model A', metrics.roc_auc_score(y_true, y_pred_prob_A))
print('AUC score from predict probabilities - Model B', metrics.roc_auc_score(y_true, y_pred_prob_B))

from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, target_names=['glaucoma', 'normal']))

# print(classification_report(y_true,y_pred,target_names=['ants', 'bees']))
# print(classification_report(originalLabels,predictedLabels,labels=labels))

from sklearn.metrics import confusion_matrix

labels = ['glaucoma', 'normal']
# labels = ['ants', 'bees']
# print(confusion_matrix(y_true,y_pred,labels))
cnf_matrx = confusion_matrix(y_true, y_pred)
print(confusion_matrix(originalLabels, predictedLabels, labels=labels))
from sklearn.metrics import roc_auc_score

#extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)

print('Model path :', PATH)
print('Test images path :', test_path)

# print('ROC score:', roc_auc_score(y_true, y_pred))
plt.figure()
plot_confusion_matrix(cnf_matrx, classes=['glaucoma', 'normal'])
# plt.show()

print('')


