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

import pickle
import numpy as np

from mtcnn.mtcnn import MTCNN
from tensorflow import keras
from matplotlib import pyplot

from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf
import time
from PIL import Image

from modules.models import RetinaFaceModel
from modules.utils import (set_memory_growth, load_yaml, draw_bbox_landm, pad_input_image, recover_pad_output)

from ultralytics import YOLO

class PredictionEngine:
    def __init__(self):
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.confidence_threshold = 0.8
        
        # Faster RCNN #
        self.model_fasterrcnn = self.get_model_instance_segmentation(3)
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model_fasterrcnn.to(torch.device('cpu'))
        checkpoint = torch.load("trained_models/fasterrcnn/checkpoint_99.pth")
        self.model_fasterrcnn.load_state_dict(checkpoint['model_state_dict'])
        self.model_fasterrcnn.eval()

        # RetinaNet #
        self.model_retinanet = torch.load('trained_models/retinanet_2epoc')
        self.model_retinanet.eval()

        # YOLOv8 #
        self.model_yolov8 = YOLO(model="runs/detect/train4/weights/best.pt")

    def FasterRCNN(self, img_path):
        image_input = Image.open(img_path).convert('RGB')
        tensor_image = self.image_transform(image_input)
        preds = self.model_fasterrcnn([tensor_image])

        boxes = preds[0]['boxes'].cpu().detach().numpy()
        labels = preds[0]['labels'].cpu().detach().numpy()
        scores = preds[0]['scores'].cpu().detach().numpy()

        results = [] 
        for box, label, score in zip(boxes, labels, scores):
            if score < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = box
            results.append(
                {
                    'box': [x1, y1, x2-x1, y2-y1],
                    'label': label,
                    'confidence': score,
                }
            )
        return results
    
    def RetinaNet(self, img_path):
        image_input = Image.open(img_path).convert('RGB')
        tensor_image = self.image_transform(image_input)
        preds = self.model_retinanet([tensor_image])

        boxes = preds[0]['boxes'].cpu().detach().numpy()
        labels = preds[0]['labels'].cpu().detach().numpy()
        scores = preds[0]['scores'].cpu().detach().numpy()

        results = [] 
        for box, label, score in zip(boxes, labels, scores):
            if score < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = box
            results.append(
                {
                    'box': [x1, y1, x2-x1, y2-y1],
                    'label': label,
                    'confidence': score,
                }
            )
        return results
    
    def YOLOv8(self, img_path):
        preds = self.model_yolov8([img_path])

        results = [] 
        for pred in preds[0].boxes:
            x1, y1, x2, y2 = pred.xyxy.cpu().detach().numpy()[0]
            results.append(
                {
                    'box': [x1, y1, x2-x1, y2-y1],
                    'label': 'label',
                    'confidence': 0.9,
                }
            )

        return results
    
    def get_model_instance_segmentation(self, num_classes):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone = None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model