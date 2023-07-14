import os
import cv2
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import torchvision
from torchvision import transforms, datasets, models
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import time
from prediction_engine import PredictionEngine

DIR_INPUT = "datasets/merged_dataset/"
DIR_IMAGES = DIR_INPUT + "images/"

class_to_int = {'background': 0, 'face_with_mask': 1, 'face_no_mask': 2}
int_to_class = {0: 'background', 1: 'face_with_mask', 2: 'face_no_mask'}

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone = None)
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
# model.to(torch.device('cpu'))
# checkpoint = torch.load("trained_models/fasterrcnn/checkpoint_99.pth")
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()
prediction = PredictionEngine()

df = pd.read_csv("datasets/merged_dataset/label.csv")
file = open("valid_dataset.txt", "r")

test_files = []

for line in file:
    test_files.append(line.strip())

file.close()

def calculate_iou(boxA, boxB):
    # Extract coordinates from the bounding boxes
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Calculate the intersection area
    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Calculate the area of both bounding boxes
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Calculate the IOU
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

    return iou

def plot_img(image_name, plot_results):
    
    fig, ax = plt.subplots(1, 2, figsize = (14, 14))
    ax = ax.flatten()
    
    bbox = df[df['name'] == image_name]
    img_path = os.path.join(DIR_IMAGES, image_name)
    
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image2 = image.copy()
    
    for idx, row in bbox.iterrows():
        x1 = row['x1']
        y1 = row['y1']
        x2 = row['x2']
        y2 = row['y2']
        label = row['classname']
        
        cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, label, (int(x1),int(y1-10)), font, 1, (255,0,0), 2)
    
    ax[0].set_title('Original Image')
    ax[0].imshow(image)

    for pr in plot_results:
        x1, y1, x2, y2 = pr['box']
        label = pr['label']
        cv2.rectangle(image2, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image2, label, (int(x1),int(y1-10)), font, 1, (255,0,0), 2)

    ax[1].set_title('Image with Bondary Box')
    ax[1].imshow(image2)

    plt.show()

confidence_threshold = 0.8
iou_threshold = 0.8
image_transform = transforms.Compose([
    transforms.ToTensor(),
])

file_test = open("test_fasterrcnn.csv", "a")
metrics = []

for index_target_name in tqdm(range(1)):
    target_name = test_files[index_target_name]
    target_name = "3642.png"

    target = df[df["name"] == target_name]

    grounds = []
    for index, row in target.iterrows():
        grounds.append(
            {
                'box': [row.x1, row.y1, row.x2, row.y2],
                'label': row.classname,
            }
        )

    results, execution_time = prediction.FasterRCNN("datasets/merged_dataset/images/"+target_name)

    plot_img(target_name, results)

    removed_grounds = []
    removed_results = []
    tp = 0
    fp=0

    for i in range(len(grounds)):
        if i in removed_grounds:
            continue
        for j in range(len(results)):
            if j in removed_results:
                continue
            iou = calculate_iou(grounds[i]['box'], results[j]['box'])
            print(iou)
            if iou > iou_threshold:
                if grounds[i]['label']==results[j]['label']:
                    tp+=1
                else:
                    fp+=1
                removed_grounds.append(i)
                removed_results.append(j)
                break

    fn = (len(grounds)-len(removed_grounds))+(len(results)-len(removed_results))
    metrics.append(f"{target_name},{tp},{fp},{fn},{confidence_threshold},{iou_threshold},{execution_time}\n")

for m in metrics:
    file_test.write(m)