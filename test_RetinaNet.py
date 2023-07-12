import albumentations as A
import torch
import torchvision
import pytorch_lightning as pl

import os
import numpy as np
from PIL import Image
from torchvision import transforms
from xml.etree.ElementTree import parse
from albumentations.pytorch import ToTensor

import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont

import math
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights

class Net(pl.LightningModule):

    def __init__(self, n_class = 4):
        super().__init__()

        self.model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
        self.num_anchors = self.model.head.classification_head.num_anchors

        self.model.head.classification_head.num_classes = n_class

        self.cls_logits = torch.nn.Conv2d(256, self.num_anchors * n_class, kernel_size = 3, stride = 1, padding = 1)
        torch.nn.init.normal_(self.cls_logits.weight, std = 0.01)  # RetinaNetClassificationHead
        torch.nn.init.constant_(self.cls_logits.bias, - math.log((1 - 0.01) / 0.01))  # RetinaNetClassificationHead
        self.model.head.classification_head.cls_logits = self.cls_logits

        for p in self.model.parameters():
            p.requires_grad = False

        for p in self.model.head.classification_head.parameters():
            p.requires_grad = True

        for p in self.model.head.regression_head.parameters():
            p.requires_grad = True

        self.model.cuda()

    def forward(self, x, t = None):
        if self.training:
            return self.model(x, t)
        else:
            return self.model(x)

    def training_step(self, batch, batch_idx):
        x, t = batch
        losses = self(x, t)
        loss = sum(losses.values())
        self.log('train_loss', loss, on_step = True, on_epoch = True, prog_bar = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        losses = self.train().forward(x, t)
        loss = sum(losses.values())
        self.log('val_loss', loss, on_step = False, on_epoch = True)

    def test_step(self, batch, batch_idx):
        x, t = batch
        losses = self.train().forward(x, t)
        loss = sum(losses.values())
        self.log('test_loss', loss, on_step = False, on_epoch = True)

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params)
        return optimizer


def visualize_results(input, output):

    mask_labels = ['BG', 'without_mask', 'with_mask', 'mask_weared_incorrect']

    image = input.permute(1, 2, 0).numpy()
    image = Image.fromarray((image * 255).astype(np.uint8))

    boxes = output['boxes'].cpu().detach().numpy()
    labels = output['labels'].cpu().detach().numpy()

    if 'scores' in output.keys():
        scores = output['scores'].cpu().detach().numpy()
        boxes = boxes[scores > 0.5]
        labels = labels[scores > 0.5]

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('fonts/NotoSansCJKjp-Bold.otf', 6)
    for box, label in zip(boxes, labels):
        # box
        draw.rectangle(box, outline = 'red')
        # label
        text = mask_labels[label]
        w, h = font.getsize(text)
        draw.rectangle([box[0], box[1], box[0] + w, box[1] + h], fill = 'red')
        draw.text((box[0], box[1]), text, font = font, fill = 'white')

    return image

# net = Net()

# net.cpu().eval()
# x, t = test2[0]
# y = net(x.unsqueeze(0))[0]

faster_rcnn_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        ])


image_input = Image.open("data/0048.jpg").convert('RGB')

        # Apply the transformation pipeline to the image
tensor_image = faster_rcnn_transform(image_input)

saved_model = torch.load('trained_models/retinanet_2epoc')
saved_model.eval()
preds = saved_model([tensor_image])
print(preds)
