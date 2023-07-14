import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from ultralytics import YOLO

class_to_int = {'background': 0, 'face_with_mask': 1, 'face_no_mask': 2}
int_to_class = {0: 'background', 1: 'face_with_mask', 2: 'face_no_mask'}

df_ground = pd.read_csv("datasets/merged_dataset/label.csv")

def plot_img(image_name, plot_results):
    
    fig, ax = plt.subplots(1, 2, figsize = (14, 14))
    ax = ax.flatten()
    
    bbox = df_ground[df_ground['name'] == image_name]
    img_path = os.path.join("datasets/merged_dataset/images/", image_name)
    
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
    
    ax[0].set_title('Original Image from Source of Truth')
    ax[0].imshow(image)

    for pr in plot_results:
        x1 = pr['box'][0]
        y1 = pr['box'][1]
        x2 = pr['box'][2]
        y2 = pr['box'][3]
        label = str(int_to_class[pr['label']])
        score = str(pr['confidence'])
        cv2.rectangle(image2, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image2, label, (int(x1),int(y1-10)), font, 1, (255,0,0), 2)
        cv2.putText(image2, score, (int(x1),int(y2+30)), font, 1, (255,0,0), 2)

    ax[1].set_title('Image with Prediction from YOLOv8')
    ax[1].imshow(image2)

    plt.show()

target_file = "4893.png"

image_transform = transforms.Compose([
    transforms.ToTensor(),
])

model = YOLO(model="runs101/detect/train/weights/epoch100.pt")

preds = model(["datasets/merged_dataset/images/"+target_file])

results = [] 
for box in preds[0].boxes:
    label_id = int(box.cls.cpu().detach().numpy()[0])+1
    score = box.conf.cpu().detach().numpy()[0]
    x1, y1, x2, y2 = box.xyxy.cpu().detach().numpy()[0]
    results.append(
        {
            'box': [x1, y1, x2-x1, y2-y1],
            'label': label_id,
            'confidence': score,
        }
    )


plot_img(target_file, results)

# model = torchvision.models.detection.retinanet_resnet50_fpn(weights=None, num_classes = 3)
# model.to(torch.device('cpu'))
# checkpoint = torch.load("trained_models/fasterrcnn/checkpoint_99.pth")
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()