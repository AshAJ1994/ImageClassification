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
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model1_RNRG/Models/scratch_model/VGG/vgg_2000_bestEpoch.pt' # resnet scratch model - 10 Epoch
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

# resnet model
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/resnet_finetuned/resnet_2000_bestEpoch.pt'
PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/resnet_ft_scratch/resnet_2000_bestEpoch.pt' # scratch model
# Model4 - 5000N + 5000G
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_FNFG/Models/resnet_finetuned/resnet_10000_8.pt'

#Unet images - CAM classification
# real images
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/UNet_Segmentation_RealImages/classif_model/Models/Resnet_finetuned/resnet_2000_bestEpoch.pt'

# fake images -  test : 500N + 500G, train: 5000N + 5000G, valid: 1000N + 1000G
# PATH = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/UNet_Segmentation/classif_model/Models/Resnet_finetuned/resnet_2000_bestEpoch.pt'

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()

def return_CAM(feature_conv, weight, class_idx):
    # generate the class -activation maps upsample to 256x256
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

# ants bees dataset - test data path
# test_path = '/home/sysadmin/Ashish/ants_bees_dataset/test/'

# real images - test path containing 78 images per class : RNFL based classification - (1000 G + 1000 N training)
test_path = '/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/test'

#fake images - 78G + 78N
# test_path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/test/'

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
test_data_loader = DataLoader(test_data, shuffle=True, batch_size=1, num_workers=0)
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
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:0,1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else "cpu")

# Load
model = torch.load(PATH)
model = model.to(device)
# params = list(model.parameters())
# weights = np.squeeze(params[-1].data.cpu().numpy())
# weight_softmax_params = list(model._modules.get('fc').parameters())
# weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
# final_layer = model._modules.get('layer4')
# activated_features = SaveFeatures(final_layer)

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

import os
class CamExtractor():
    """
        Extracts cam features from the model
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_name, module in self.model._modules.items():
            print(module_name)
            if module_name == 'fc':
                return conv_output, x
            x = module(x)  # Forward
            #print(module_name, module)
            #resnet
            if module_name == self.target_layer:
                print('True')
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
            #VGG
            elif module_name == 'features':
                print('vgg cam hook register to be done')
                x.register_hook(self.save_gradient)
                conv_output = x
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.fc(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_index=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        chk = imgConversion_transform(conv_output.cpu().squeeze())
        if target_index is None:
            target_index = np.argmax(model_output.cpu().data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().to(device)
        one_hot_output[0][target_index] = 1
        # Zero grads
        self.model.fc.zero_grad()
        # self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.cpu().data.numpy()[0]
        # Get convolution outputs
        target = conv_output.cpu().data.numpy()[0]
        # Get weights from gradients
        # Take averages for each gradient
        weights = np.mean(guided_gradients, axis=(1, 2))
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = cv2.resize(cam, (224, 224))
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) -
                                     np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        return cam

def save_class_activation_on_image(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('./results'):
        os.makedirs('./results')
    # Grayscale activation map
    path_to_file = os.path.join('./results', file_name + '_Cam_Grayscale.jpg')
    cv2.imwrite(path_to_file, activation_map)
    # Heatmap of activation map
    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_HSV)
    path_to_file = os.path.join('./results', file_name + '_Cam_Heatmap.jpg')
    cv2.imwrite(path_to_file, activation_heatmap)
    # Heatmap on picture
    org_img = cv2.resize(org_img, (224, 224))
    img_with_heatmap = np.float32(activation_heatmap) + np.float32(org_img)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    path_to_file = os.path.join('./results', file_name + '_Cam_On_Image.jpg')
    cv2.imwrite(path_to_file, np.uint8(255 * img_with_heatmap))

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

import torch
from torchcam.cams import CAM, GradCAM, SmoothGradCAMpp

# with torch.no_grad():
for images,labels,filenames in test_data_loader:
    images = images.to(device)
    labels = labels.to(device)

    images.requires_grad = True

    prediction = model(images)
    pred_probabilities = F.softmax(prediction).data.cpu().squeeze()
    pred_probabilities_sigmoid = F.sigmoid(prediction).data.cpu().squeeze()
    class_idx = topk(pred_probabilities, 1)[1].int()
    l = str(float("{:.2f}".format(torch.max(pred_probabilities))))
    buf = 'predicted class :' + class_names[class_idx] + '   probability : ' + l + '  actual class : ' + class_names[
        int(labels[0])]

    # Grad cam
    grad_cam = GradCam(model, target_layer='layer4') # for resnet model
    # grad_cam = GradCam(model, target_layer=model.features[-1]) # for vgg11 model (not yet working) - switch to GradCAM_Visualization.py script
    # Generate cam mask

    cam = grad_cam.generate_cam(images)
    # Save mask
    # save_class_activation_on_image(images, cam, 'Check')

    img = cv2.imread(filename=filenames[0])
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
    path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/test_CAM_scratch/resnet/FonR/' # resnet fake model on real images

    #UNET images
    # real images
    # path = test_path + 'test_CAM/'
    # fake images
    # path = test_path + 'test_CAM_FonF/'
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/UNet_Segmentation/classif_model/test/test_CAM_FonR/'
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/test_CAM_vgg_ft/' # VGG - 1000n + 1000g
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/Models/test_CAM_resnet_ft/' # resnet 1000 n + 1000g
    # path = '/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_FNFG/Models/test_CAM_resnetresults/' # resnet 5000 N + 5000 G

    saveCAMImage(img, cam, resultText=buf, saveDirPath=path)

    # using pytorch official doc method for calculating loss / correct predictions
    outputs = model(images)
    prob, preds = torch.max(outputs, 1)
    # print('preds',preds)
    y_true += list(labels.cpu().numpy())
    y_pred += list(preds.cpu().numpy())
    total += labels.size(0)
    correct += (preds == labels.data).sum().item()

    for j in range(images.size()[0]):
        originalLabels.append(class_names[labels[j]])
        predictedLabels.append(class_names[preds[j]])

print('Test accuracy using Pytorch official doc method : {:0.2f} %'.format(100 * (correct / total)))

# evaluation metrics
print(metrics.accuracy_score(y_true, y_pred))
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

# print('ROC score:', roc_auc_score(y_true, y_pred))
plt.figure()
plot_confusion_matrix(cnf_matrx, classes=['glaucoma', 'normal'])
plt.show()

print('')


