import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch
import torchvision
from torchvision import transforms

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

    ax[1].set_title('Image with Prediction from RetinaNet')
    ax[1].imshow(image2)

    plt.show()

target_file = "2229.png"

image_transform = transforms.Compose([
    transforms.ToTensor(),
])

model = torchvision.models.detection.retinanet_resnet50_fpn(weights=None, num_classes = 3)
model.to(torch.device('cpu'))
checkpoint = torch.load("trained_models/retinanet/checkpoint_retinanet_99.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

image_input = Image.open("datasets/merged_dataset/images/"+target_file).convert('RGB')
tensor_image = image_transform(image_input)
preds = model([tensor_image])
image_input.close()

boxes = preds[0]['boxes'].cpu().detach().numpy()
labels = preds[0]['labels'].cpu().detach().numpy()
scores = preds[0]['scores'].cpu().detach().numpy()

results = [] 
for box, label, score in zip(boxes, labels, scores):
    if score < 0.8:
        continue
    x1, y1, x2, y2 = box
    results.append(
        {
            'box': [x1, y1, x2, y2],
            'label': label,
            'confidence': score,
        }
    )

plot_img(target_file, results)

# model = torchvision.models.detection.retinanet_resnet50_fpn(weights=None, num_classes = 3)
# model.to(torch.device('cpu'))
# checkpoint = torch.load("trained_models/fasterrcnn/checkpoint_99.pth")
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()