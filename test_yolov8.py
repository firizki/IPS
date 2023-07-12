import numpy as np
import pandas as pd
import os
import random
import shutil
import glob
import yaml
import torch
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import cv2

from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import patches
from pathlib import Path
from sklearn.model_selection import train_test_split

import ultralytics
from ultralytics import YOLO
ultralytics.checks()

model = YOLO(model="runs/detect/train4/weights/best.pt")

results = model(["data/utm_face_example.jpg"])

# print(results)

for box in results[0].boxes:
    # boxes = box.boxes  # Boxes object for bbox outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Class probabilities for classification outputs
    x1, y1, x2, y2 = box.xyxy.cpu().detach().numpy()[0]
    print(x1, y1, x2, y2)

# predicted_image = Image.open("runs/detect/predict/image0.jpg")
# plt.figure(figsize=(10,10))
# plt.imshow(predicted_image)
# plt.title("Prediction")
# plt.axis(False)
# plt.show()