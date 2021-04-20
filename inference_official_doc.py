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

# PATH = "glauc_classif_VGG16_100Samples_Training.pt"
# PATH = "glauc_classif_VGG16_100(2)Samples_Training.pt"
# PATH = "glauc_classif_VGG16_100(3)Samples_Training.pt"
# PATH = "glauc_classif_VGG16_200(2)Samples_Training.pt"
# PATH = "glauc_classif_VGG16_200(3)Samples_Training.pt"
# PATH = "glauc_classif_VGG16_400(2)Samples_Training.pt"
# PATH = 'glauc_classif_VGG16_250Samples_Training.pt'
# PATH = "glauc_classif_VGG16_300sampleTraining.pt"
# PATH = 'glauc_classif_VGG16_500Samples_Training.pt'
# PATH = "glauc_classif_VGG16_800Samples_Training.pt"
# PATH = "glauc_classif_VGG16_AllSamples_Training.pt"
# PATH = "glauc_classif_VGG16_All(2)Samples_Training.pt"

# PATH = "200Glau_training.pt"
# PATH = "300Glau_training.pt"
# PATH = "400Glau_training.pt"
# PATH = "600Glau_training.pt"
# PATH = "800Glau_training.pt"
# PATH = "1000Glau_training.pt"

# PATH = "ChineseFull_1000Glauc_model.pt"
# PATH = "Chinese_Fake_model.pt"
# PATH = "FullModel_Classification_(Dr.jacOnlyGlaucomaData).pt"
# PATH = "FullModel_Classification.pt"

# PATH = "All_Classification_model_v2.pt"
# PATH = "best_model.pt"

# ----------------------------------------------------
# oct21st meeting model's
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
# PATH = 'latest_Resnet_realimages.pt'

# PATH = 'Consolidated_Resnet_Model_RealImages.pt'
# PATH = 'Consolidated_Resnet_Model_FakeImages.pt'
# PATH = 'Consolidated_Resnet_Model_FakeImages_20000.pt'
# PATH = 'Latest_GAN_Resnet_6.pt'
# PATH = 'Latest_GAN_Resnet_14.pt'
# PATH = 'latestGAN_40000.pt'

# PATH = 'Hope_8.pt'
# PATH = 'Hope_6.pt'
# PATH = 'hope.pt'
# PATH = 'Hope_Epoch30_4.pt'
# PATH = 'Hope_Epoch30_28.pt'
# PATH = 'Inception_28.pt'
# PATH = 'Hope_Epoch30_20.pt'
# PATH = 'Mixed_Resnet_Model.pt'
# PATH = 'inception_originalImages.pt'
# PATH = 'resnet_originalImages.pt'
# PATH = 'resnet_3200_FakeImages_Epochs10.pt'
# PATH = 'resnet_6000_FakeImages_Epochs10.pt'
# PATH = 'resnet_6000(v2)_FakeImages_Epoch:8.pt' # lr = 0.0001
# PATH = 'resnet_selectedImages_GAN_10.pt'

# PATH = 'resnet1lkhTrainImages_selectedImages_GAN_pt'
# PATH = 'resnet1lkhTrainImages_selectedImages_GAN_0.pt'
# PATH = 'resnet1lkhTrainImages_selectedImages_GAN_2.pt'
# PATH = 'resnet1lkhTrainImages_selectedImages_GAN_4.pt'
# PATH = 'resnet1lkhTrainImages_selectedImages_GAN_6.pt'
# PATH = 'resnet1lkhTrainImages_selectedImages_GAN_28.pt'
# PATH = 'resnet_1lkh_0.0001lr/resnet1lkhTrainImages_selectedImages_GAN_8.pt'
# PATH = 'resnet1lkhTrainImages_selectedImages_GAN_10.pt'
# PATH = '1lkhImage_v2.0_0.pt'
# PATH = '1lkhImage_v2.0_2.pt'
# PATH = '1lkhImage_v2.0_4.pt'
# PATH = '1lkhImage_v2.0_Resnet.pt'

#validation from Scratch

# ***** included 50 test images from 1590 , 1147 used in GAN training *******
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/40000FakeImages_training/Models/PytorchMethod1(Generic)/resnet_80000_4.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/40000FakeImages_training/Models/PytorchMethod2(Resnet)/Resnet_80000_bestEpoch.pt'

# PATH = '/home/sysadmin/Ashish/40000_Classification_OnlyFake/Models/PytorchMethod1(Generic)/resnet_80000_bestEpoch.pt'

# PATH = '/home/sysadmin/Ashish_PGAN_Validation/40000FakeImages+realNormalImages_training/Models/PytorchMethod2(Resnet)/Resnet_82622_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/40000FakeImages+realNormalImages_training/Models/PytorchMethod1(Generic)/resnet_82622_bestEpoch.pt'

# ******* without 50 test images in GAN training ********
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/50000FakeImages+realImages_traiining/Models/PytorchMethod2(Resnet)/PytorchMethod2(Resnet)Resnet_102637_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/50000FakeImages+realImages_traiining/Models/PytorchMethod1(Generic)/PytorchMethod1(Generic)resnet_102637_bestEpoch.pt'

# Real images path - 900 training data for normal and glaucoma
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/RealImages(1590,523+627)_Training/Models/PytorchMethod1(Generic)/resnet_1800_bestEpoch.pt' #generic
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/RealImages(1590,523+627)_Training/Models/PytorchMethod2(Resnet)/Resnet_1800_bestEpoch.pt' #resnet only

# 50000 fake images + 900 real images for each class
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/50000Fake+900RealImages_Training/Models/PytorchMethod1(Generic)/resnet_101800_bestEpoch.pt' #generic
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/50000Fake+900RealImages_Training/Models/PytorchMethod2(Resnet)/Resnet_101800_bestEpoch.pt' #resnet only

# 50000 fake images only (from latest GAN method for Nov4th meeting : without test images)-- no real images included for classification
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/50000_FakeImagesOnly_training/Models/PytorchMethod1(Generic)/PytorchMethod1(Generic)resnet_100000_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/50000_FakeImagesOnly_training/Models/PytorchMethod2(Resnet)/PytorchMethod2(Resnet)Resnet_100000_bestEpoch.pt'

# Epochs - 20
# Real images path - 1100 training data for normal and glaucoma , 100 - valid, 72 - test each : RNFL based data split for Glaucoma
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA/Models/Epochs_20/PytorchMethod1(Generic)/resnet_2200_bestEpoch.pt' #generic
# Epochs - 10
# Real images path - 1100 training data for normal and glaucoma , 100 - valid, 72 - test each : RNFL based data split for Glaucoma
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA/Models/Epochs_10/PytorchMethod1(Generic)/resnet_2200_bestEpoch.pt' #generic

# Fake images Only path - 50000 training data for normal and glaucoma (trained using 1100G + 1100N real data : RNFL based data split for Glaucoma)
# Epochs -10
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FinalizedGAN_Severity_GANImages/Models/GenericModel/GenericModelresnet_100000_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FinalizedGAN_Severity_GANImages/Models/GenericModel/GenericModelresnet_100000_6.pt'

#Fake images + real images - 50000N + 1100N, 50000G + 1100G (trained using 1100G + 1100N real data : RNFL based data split for Glaucoma)
# epochs - 10
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FinalizedGAN_Severity_GANImages+RealImages(1100)/Models/GenericModel/GenericModelresnet_102200_bestEpoch.pt'
# epochs =20
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FinalizedGAN_Severity_GANImages+RealImages(1100)/Models/GenericModelresnet_102200_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FinalizedGAN_Severity_GANImages+RealImages(1100)/Models/GenericModelresnet_102200_bestEpoch.pt'
# ******** best accuracy - epoch 6  below:
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FinalizedGAN_Severity_GANImages+RealImages(1100)/Models/GenericModel/GenericModelresnet_102200_6.pt'
# **********************************************
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FinalizedGAN_Severity_GANImages+RealImages(1100)/Models/GenericModel/GenericModelresnet_102200_4.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FinalizedGAN_Severity_GANImages+RealImages(1100)/Models/GenericModel/GenericModelresnet_102200_2.pt'

#Fake images + real images - 50000N + 1000N, 50000G + 1000G (trained using 1000G + 1000N real data : Severity based data split for Glaucoma)
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FInalized_GAN_basedOnSeverity_Fake+Real(1000)/Models/GenericModel/resnet_102000_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FInalized_GAN_basedOnSeverity_Fake+Real(1000)/Models/GenericModel/resnet_102000_8.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FInalized_GAN_basedOnSeverity_Fake+Real(1000)/Models/GenericModel/resnet_102000_12.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FInalized_GAN_basedOnSeverity_Fake+Real(1000)/Models/GenericModel/resnet_102000_18.pt'

#Fake images only - 50000N, 50000G (trained using 1000G + 1000N real data : Severity based data split for Glaucoma)
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FinalizedGAN_Severity_GANImages/Models/GenericModelresnet_100000_6.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FinalizedGAN_Severity_GANImages/Models/GenericModelresnet_100000_16.pt'

#Fake images only -(five thousand ) 5000 N, 5000 G (trained using 1000G + 1000N real data : Severity based data split for Glaucoma)
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/Models/GenericModel/resnet_10000_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/Models/GenericModel/resnet_10000_4.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/Models/GenericModel/resnet_10000_bestEpoch.pt'

#Fake images(five thousand) + real images(1000) - 5000 N + 1000 N , 5000 G + 1000 G (trained using 1000G + 1000N real data : Severity based data split for Glaucoma)
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000Fake+1000Real/Models/GenericModel/resnet_12000_10.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000Fake+1000Real/Models/GenericModel/resnet_12000_6.pt'

# PATH = '/home/sysadmin/Ashish_PGAN_Validation/GAN_Severity_Images_5000/Models/Check/resnet_10000_bestEpoch.pt'

# Real images - 1000G , 1000N : based on severity
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/Models/PytorchMethod1(Generic)/ResnetBased/resnet_2000_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/Models/PytorchMethod2(Resnet)/Resnet_2000_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/Models/PytorchMethod1(Generic)/VGGBased/vgg_2000_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/Models/PytorchMethod1(Generic)/VGGBased/vgg_2000_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/Models/PytorchMethod1(Generic)/VGGBased/vgg_2000_10.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/Models/PytorchMethod1(Generic)/resnet_2000_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/Models/PytorchMethod1(Generic)/resnet_2000_10.pt'

# Flattened real images - 1000G + 1000N, 100G+100N valid, 78G+78N : test based on severity
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_Severity_FlattemedImages/Models/PytorchMethod1(Generic)/resnet_2000_bestEpoch.pt'

# Flattened fake images - 5000G + 5000N, 1000G + 1000N valid, 100G + 100N test : based on severity
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Flattened_GAN_FakeImages_Results/Models/PytorchMethod1(Generic)/resnet_10000_bestEpoch.pt'

# CAM visualization - real images (1000N + 1000G)
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/Models/CAM/Resnet_2000_bestEpoch.pt'

# CAM visualization - FLATTENED real images (1000N + 1000G)
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_Severity_FlattemedImages/Models/PytorchMethod1(Generic)/resnet_2000_bestEpoch.pt'

# CAM visualization trial - Ants Bees dataset
# PATH = '/home/sysadmin/Ashish/ants_bees_dataset/Models/Resnet/Resnet_224_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish/ants_bees_dataset/Models/BW_Resnet/Resnet_224_bestEpoch.pt'
# PATH = '/home/sysadmin/Ashish/ants_bees_dataset/Models/BW_Resnet/Resnet_224_bestEpoch.pt'

# def set_parameter_requires_grad(model, feature_extracting):
#     if feature_extracting:
#         for param in model.parameters():
#             param.requires_grad = False
# from torchvision import models
# import torch.nn as nn
# model_ft = models.resnet18(pretrained=True)
# set_parameter_requires_grad(model_ft, False)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 2)
# input_size = 224

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()

def return_CAM(feature_conv, weight, class_idx):
    # generate the class-activation maps upsample to 256x256
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    # for idx in class_idx:
    # print(weight[class_idx])
    beforeDot = feature_conv.reshape((nc, h*w))
    # cam = np.matmul(weight[idx], beforeDot)
    cam = weight[class_idx].dot(beforeDot)
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

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


# test_path = '/home/sysadmin/Ashish/Chinese_Glaucoma_ClassificationData/test_allSamples'
# test_path = '/home/sysadmin/Ashish/Chinese_Glaucoma_ClassificationData/test_chineseOnly'
# test_path = '/home/sysadmin/Ashish/Chinese_Glaucoma_ClassificationData/test'
# test_path = '/home/sysadmin/Ashish/Chinese_Glaucoma_ClassificationData/test_fake_chineseOnly'
# test_path = '/home/sysadmin/Ashish/all_classification_test'
# test_path = '/home/sysadmin/Ashish/All_Classification'
# test_path = '/home/sysadmin/Ashish/Chinese_Glaucoma_ClassificationData/train'
# test_path = '/home/sysadmin/Ashish/Chinese_Glaucoma_ClassificationData/test'
# oct21st

# test_path = '/home/sysadmin/Ashish/Oct21/Chinese_original_images/test'
# test_path = '/home/sysadmin/Ashish/Oct21/Chinese_fake_images/test'
# test_path = '/home/sysadmin/Ashish/Oct21/Chinese_fake_images/test(source)'
# test_path = '/home/sysadmin/Ashish/Oct21/ImprovedGAN'
# test_path = '/home/sysadmin/Ashish/Oct21/Chinese_original_images/train'
# test_path = '/home/sysadmin/Ashish/40000_Classification/test'


# test_path = '/home/sysadmin/Ashish/All_Classification/valid'
# test_path = '/home/sysadmin/Ashish/40000_Classification/train'
# test_path = '/home/sysadmin/Ashish/20000_Classification/test'

# classification
# test_path = '/home/sysadmin/Ashish/Latest_GANTraining_SplitImages/test'
# test_path = '/home/sysadmin/Ashish/Latest_GANTraining_SplitImages/Normal/Gan_train_images'
# test_path = '/home/sysadmin/Ashish/latest_GAN_generatedImages/test'
# test_path = '/home/sysadmin/Ashish/latest_GAN_generatedImages/train'
# test_path = '/home/sysadmin/Ashish/latest_GAN_v2.0/train'
# test_path = '/home/sysadmin/Ashish/Latest_GANTraining_SplitImages/Check_Results/test' #main test location (50-50)
# test_path = '//home/sysadmin/Ashish/20000_Classification/test'

# test_path = '/home/sysadmin/Ashish/Latest_GANTraining_SplitImages/Check_Results/train'
# test_path = '/home/sysadmin/Ashish/Chinese_Classification_CrossValidation/train'

# test_path = '/home/sysadmin/Ashish/AllClassification-FakeImages/test'
# test_path = '/home/sysadmin/Ashish/LatestGAN_Fake_ClassificationModel/latestGAN_v4.0_10000images_UpdatedGlaucoma_images/test'

# new test path - unseen image
# test_path = '/home/sysadmin/Desktop/newTestPath_pgan/test_50'
# test_path = '/home/sysadmin/Desktop/newTestPath_pgan/test_50_v2'
# test_path = '/home/sysadmin/Desktop/newTestPath_pgan/test_extra_chinese_50'

# real images - test path containing 97 images per class : '/home/sysadmin/Ashish_PGAN_Validation/RealImages(1590,523+627)_Training/test'
# test_path = '/home/sysadmin/Ashish_PGAN_Validation/RealImages(1590,523+627)_Training/test'

# real images - test path containing 72 images per class : RNFL based classification - (1100 G + 1100 N training)
# test_path = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA/test'

# real images - test path containing 78 images per class : RNFL based classification - (1000 G + 1000 N training)
test_path = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/test'

# FLATTENED real images - test path containing FLATTENED 78 images per class : RNFL based classification - (1000 G + 1000 N training)
# test_path = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_Severity_FlattemedImages/test/'

# ants bees dataset - test data path
# test_path = '/home/sysadmin/Ashish/ants_bees_dataset/test/'

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

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:0,1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else "cpu")

# Load
model = torch.load(PATH)
params = list(model.parameters())
weights = np.squeeze(params[-1].data.cpu().numpy())
weight_softmax_params = list(model._modules.get('fc').parameters())
weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
final_layer = model._modules.get('layer4')
activated_features = SaveFeatures(final_layer)

# model = model_ft.load_state_dict(PATH)
model.eval()
was_training = model.training

class_names = ('glaucoma', 'normal')
# class_names = ('ants', 'bees')

correct = 0
total = 0
y_true = []
y_pred = []
originalLabels = []
predictedLabels = []
#
# for images,labels,filenames in test_data_loader:
#     images = images.to(device)
#     labels = labels.to(device)
#     images.requires_grad = True
#     outputs = model(images)
#     outputs_idx = outputs.argmax()
#     output_max = outputs[0, outputs_idx]
#     output_max.backward()
#
#     saliency,_ = torch.max(images.grad.data.abs(), dim=1)
#     saliency = saliency.reshape(224,224)
#
#     # Reshape the image
#     images = images.reshape(-1, 224, 224)
#
#     # Visualize the image and the saliency map
#     fig, ax = plt.subplots(1, 2)
#     ax[0].imshow(images.cpu().detach().numpy().transpose(1, 2, 0))
#     ax[0].axis('off')
#     ax[1].imshow(saliency.cpu(), cmap='jet')
#     ax[1].axis('off')
#     plt.tight_layout()
#     fig.suptitle('The Image and Its Saliency Map')
#     plt.show()
#
#     print('')


with torch.no_grad():
    for images,labels,filenames in test_data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # ***************** CAM check *************
        # prediction = model(images)
        # pred_probabilities = F.softmax(prediction).data.cpu().squeeze()
        # activated_features.remove()
        # class_idx = topk(pred_probabilities, 1)[1].int()
        # # class_idx = topk(pred_probabilities, 1)
        # l = str(float("{:.2f}".format(torch.max(pred_probabilities))))
        # buf = 'predicted class :' + class_names[class_idx] + '   probability : ' + l + '  actual class : ' + class_names[int(labels[0])]
        # # print('class_idx', class_idx)
        # overlay = return_CAM(activated_features.features, weight_softmax, class_idx)
        # # plt.imshow(overlay[0], alpha=0.5, cmap='jet')
        # # plt.show()
        #
        # tnsr = images.data.cpu()
        # # disp_tnsr = display_transform(images[0])
        # chk = display_transform_flattenedImage(overlay[0])
        # # plt.xlabel('Predicted Class : ', str(class_names[class_idx]))
        # # imshow(tnsr[0])
        # # # plt.imshow(images[0])
        # # plt.xlabel(buf)
        # # plt.imshow(skimage.transform.resize(overlay[0], images[0].shape[1:3]), alpha=0.5, cmap='jet')
        # # plt.show()
        #
        # img = cv2.imread(filename=filenames[0])
        # height, width, _ = img.shape
        # heatmap = cv2.applyColorMap(cv2.resize(overlay[0], (width, height)), cv2.COLORMAP_JET)
        # result = heatmap * 0.5 + img * 0.5
        #
        # combine = np.concatenate([img, result],1)
        # p = Path(filenames[0])
        # # saveName = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/test_CAM_combined/' + p.name # for real images glaucoma vs normal classification
        # # saveName = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_Severity_FlattemedImages/test_CAM_combined/' + p.name # for flattened real images - G vs N
        # saveName = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/test_B&W_CAM_combined/' + p.name
        # # saveName = '/home/sysadmin/Ashish/ants_bees_dataset/test_CAM_combined/' + p.name # for ants vs bees classification
        #
        # # font
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # # org
        # org = (10, 10)
        # # org = (10, 80)
        # # fontScale
        # fontScale = 0.4
        # # Blue color in BGR
        # color = (255, 255, 255)
        # # Line thickness of 2 px
        # thickness = 1
        # cv2.putText(combine, buf, org=org, color=color, fontFace=font, fontScale=fontScale, thickness=thickness)
        # cv2.imwrite(saveName, combine)
        # ****************************************************

        # using pytorch official doc method for calculating loss / correct predictions
        prob, preds = torch.max(outputs, 1)
        # print('preds',preds)
        y_true += list(labels.cpu().numpy())
        y_pred += list(preds.cpu().numpy())
        total += labels.size(0)
        correct += (preds == labels.data).sum().item()

        for j in range(images.size()[0]):
            originalLabels.append(class_names[labels[j]])
            predictedLabels.append(class_names[preds[j]])

    print('Test accuracy using Pytorch official doc method : {:0.2f} %'.format(100* (correct/total)))

#evaluation metrics
print(metrics.accuracy_score(y_true,y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_true,y_pred,target_names=['glaucoma', 'normal']))
# print(classification_report(y_true,y_pred,target_names=['ants', 'bees']))
# print(classification_report(originalLabels,predictedLabels,labels=labels))

from sklearn.metrics import confusion_matrix
labels = ['glaucoma','normal']
# labels = ['ants', 'bees']
# print(confusion_matrix(y_true,y_pred,labels))
print(confusion_matrix(originalLabels,predictedLabels,labels=labels))
from sklearn.metrics import roc_auc_score
print('ROC score:', roc_auc_score(y_true,y_pred))

# correct_count, all_count = 0, 0
# images_so_far = 0
# for images,labels in test_data_loader:
#     for i in range(len(labels)):
#         images = images.to(device)
#         labels = labels.to(device)
#
#         outputs = model(images)
#         # Convert to probabilities
#         ps = torch.exp(outputs)
#         probab = list(ps.cpu()[0])
#         pred_label = probab.index(max(probab))
#         true_label = labels.cpu()[i]
#
#         # images_so_far += 1
#         # ax = plt.subplot(10 // 2, 2, images_so_far)
#         # ax.axis('off')
#         # print('pred:{} org:{}'.format(class_names[preds[i]], class_names[labels[i]]))
#         # imshow(images.cpu().data[i])
#         if(true_label == pred_label):
#             correct_count += 1
#         all_count += 1
#
# print("Number Of Images Tested =", all_count)
# print("\nModel Accuracy =", (100* (correct_count/all_count)))


print('')


