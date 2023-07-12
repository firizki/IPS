# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from bs4 import BeautifulSoup
import torchvision
from torchvision import transforms, datasets, models
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.patches as patches

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

model = get_model_instance_segmentation(3)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

### TRAIN MODEL SECTION ###
num_epochs = 25
model.to(device)

model.load_state_dict(torch.load("trained_models/fasterrcnn_"+str(num_epochs)+"epoc"))
model.eval()

image_input = Image.open("datasets/wobotintelligence_FaceMaskDataset/Medical mask/Medical mask/Medical Mask/images/0002.png")

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])

# Apply the transformation pipeline to the image
tensor_image = transform(image_input).cuda()
preds = model([tensor_image])
print(preds)


# for imgs, annotations in data_loader:
#         imgs = list(img.to(device) for img in imgs)
#         annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
#         break

# model.eval()
# preds = model(imgs)
# print(type(imgs[0]))
# print(len(preds))

def plot_image(img_tensor, annotation):
    
    fig,ax = plt.subplots(1)
    img = img_tensor.cpu().data

    # Display the image
    ax.imshow(img.permute(1, 2, 0))

    rect = patches.Rectangle((68, 42), 37, 27, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    boxes = annotation['boxes'].cpu().detach().numpy()
    labels = annotation['labels'].cpu().detach().numpy()
    scores = annotation['scores'].cpu().detach().numpy()

    results = []
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        results.append(
            {
                'box': [x1, y1, x2-x1, y2-y1],
                'label': label,
                'confidence': score,
            }
        )
    print(results)
    
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box.cpu().data
        xmin = xmin.item()
        ymin = ymin.item()
        xmax = xmax.item()
        ymax = ymax.item()

        # Create a Rectangle patch
        rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

print("Prediction")
plot_image(tensor_image, preds[0])
# print("Target")
# plot_image(imgs[3], annotations[3])