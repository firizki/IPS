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

class PredictionEngine:
    def __init__(self):
        self.detector_mtcnn = MTCNN()

        # Init RetinaFace
        cfg_path = './configs/retinaface_res50.yaml'
        gpu = '0'
        iou_th = 0.4
        score_th = 0.5
        
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

        logger = tf.get_logger()
        logger.disabled = True
        logger.setLevel(logging.FATAL)
        set_memory_growth()

        self.cfg = load_yaml(cfg_path)

        # define network
        self.model_retinaface = RetinaFaceModel(self.cfg, training=False, iou_th=iou_th,
                                score_th=score_th)

        # load checkpoint
        checkpoint_dir = './checkpoints/' + self.cfg['sub_name']
        checkpoint = tf.train.Checkpoint(model=self.model_retinaface)
        if tf.train.latest_checkpoint(checkpoint_dir):
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
            print("[*] load ckpt from {}.".format(
                tf.train.latest_checkpoint(checkpoint_dir)))
        else:
            print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
            exit()

        # Faster RCNN #
        self.model_fasterrcnn = self.get_model_instance_segmentation(24)
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model_fasterrcnn.to(torch.device('cpu'))

        # self.model_fasterrcnn.load_state_dict(torch.load("trained_models/fasterrcnn_25epoc"))
        checkpoint = torch.load("trained_models/checkpoint.pth")
        self.model_fasterrcnn.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        self.model_fasterrcnn.eval()
        self.faster_rcnn_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        ])

    def mtcnn(self, image):
        pixels = pyplot.imread(image)
        return self.detector_mtcnn.detect_faces(pixels)

    def retinaface(self, img_path):
        down_scale_factor = 1.0

        if not os.path.exists(img_path):
            print(f"cannot find image path from {img_path}")
            exit()

        print("[*] Processing on single image {}".format(img_path))

        img_raw = cv2.imread(img_path)
        img_height, img_width, _ = img_raw.shape
        img = np.float32(img_raw.copy())

        if down_scale_factor < 1.0:
            img = cv2.resize(img, (0, 0), fx=down_scale_factor,
                                fy=down_scale_factor,
                                interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # pad input image to avoid unmatched shape problem
        img, pad_params = pad_input_image(img, max_steps=max(self.cfg['steps']))

        # run model
        outputs = self.model_retinaface(img[np.newaxis, ...]).numpy()

        # recover padding effect
        outputs = recover_pad_output(outputs, pad_params)

        results = []

        for idx in range(len(outputs)):
            ann = outputs[idx]
            x1, y1, x2, y2 = int(ann[0] * img_width), int(ann[1] * img_height), \
                     int(ann[2] * img_width), int(ann[3] * img_height)
            results.append(
                {
                    'box': [x1, y1, x2-x1, y2-y1],
                    'confidence': ann[15],
                    'keypoints': {
                        'left_eye': (int(ann[4] * img_width), int(ann[5] * img_height)),
                        'right_eye': (int(ann[6] * img_width), int(ann[7] * img_height)),
                        'nose': (int(ann[8] * img_width), int(ann[9] * img_height)),
                        'mouth_left': (int(ann[10] * img_width), int(ann[11] * img_height)),
                        'mouth_right': (int(ann[12] * img_width), int(ann[13] * img_height))
                    }
                }
            )

        return results

    def FasterRCNN(self, img_path):
        image_input = Image.open(img_path).convert('RGB')

        # Apply the transformation pipeline to the image
        tensor_image = self.faster_rcnn_transform(image_input)
        preds = self.model_fasterrcnn([tensor_image])
        print(preds)
        results = []
        for box in preds[0]["boxes"]:
            xmin, ymin, xmax, ymax = box.cpu().data
            x1 = xmin.item()
            y1 = ymin.item()
            x2 = xmax.item()
            y2 = ymax.item()

            results.append(
                {
                    'box': [x1, y1, x2-x1, y2-y1],
                    'confidence': 0.99,
                }
            )
        return results
    
    def get_model_instance_segmentation(self, num_classes):
        # # load an instance segmentation model pre-trained pre-trained on COCO
        # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        # # get number of input features for the classifier
        # in_features = model.roi_heads.box_predictor.cls_score.in_features
        # # replace the pre-trained head with a new one
        # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone = None)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model