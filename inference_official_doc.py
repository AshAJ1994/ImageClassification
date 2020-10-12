import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
# calculate accuracy
from sklearn import metrics
from collections import Counter

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

PATH = "All_Classification_model_v2.pt"

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
test_path = '/home/sysadmin/Ashish/all_classification_test'

test_data = datasets.ImageFolder(root=test_path, transform=transforms.Compose([
                                      transforms.Grayscale(num_output_channels=3),
                                      transforms.Resize((224, 224)),
                                      # transforms.ToPILImage(),
                                      transforms.ToTensor(),
                                      # transforms.Normalize((0.5,), (0.5,),(0.5,)),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
]))
print(len(test_data))
test_data_loader = DataLoader(test_data, shuffle=True, batch_size=10, num_workers=0)
print(dict(Counter(test_data_loader.dataset.targets)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load
model = torch.load(PATH)
model.eval()
was_training = model.training
fig = plt.figure()
class_names = ('glaucoma', 'normal')

correct = 0
total = 0
y_true = []
y_pred = []
for images,labels in test_data_loader:
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)

    # using pytorch official doc method for calculating loss / correct predictions
    _, preds = torch.max(outputs, 1)
    y_true += list(labels.cpu().numpy())
    y_pred += list(preds.cpu().numpy())
    total += labels.size(0)
    correct += (preds == labels.data).sum().item()

print('Test accuracy using Pytorch official doc method : {:0.2f} %'.format(100* (correct/total)))

#evaluation metrics
print(metrics.accuracy_score(y_true,y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_true,y_pred))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_true,y_pred))
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