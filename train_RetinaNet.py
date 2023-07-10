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

class MaskDataset(torch.utils.data.Dataset):

    def __init__(self, root):
        self.root = root
        self.mask_labels = ['BG', 'without_mask', 'with_mask', 'mask_weared_incorrect']
        self.imgs = list(sorted(os.listdir('datasets/andrewmvd_FaceMaskDataset/images/')))
        # Add for data augmentation
        bbox_params = A.BboxParams(format = 'pascal_voc', label_fields = ['class_labels'])
        self.transform = A.Compose([
            A.HorizontalFlip(p = 0.1),
            A.VerticalFlip(p = 0.1),
            A.RandomBrightnessContrast(p = 0.1),
            ToTensor()
          ], bbox_params = bbox_params)

    def __getitem__(self, idx):
        data = 'maksssksksss' + str(idx)

        image_path = f'{self.root}/images/{data}.png'
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        anno_path = f'{self.root}/annotations/{data}.xml'
        parser = parse(anno_path)
        labels, boxes = [], []
        for obj in parser.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox.find('xmin').text != bndbox.find('xmax').text: # Add
                box = [int(tag.text) for tag in obj.find('bndbox')]
                label = obj.find('name').text
                label = self.mask_labels.index(label)
                boxes.append(box)
                labels.append(label)
                
        transformed = self.transform(image = image, bboxes = boxes, class_labels = labels)
        image = transformed['image']
        boxes = torch.tensor(transformed['bboxes'], dtype = torch.float32)
        labels = torch.tensor(transformed['class_labels'], dtype = torch.int64)
        
        # Add for estimation of mean Average Precisions
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((len(parser.findall('object')),), dtype = torch.int64)

        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id, 'area': area, 'iscrowd': iscrowd}

        return image, target


    def __len__(self):
        return len(self.imgs)
    
class MaskDatasetNoAug(torch.utils.data.Dataset):

    def __init__(self, root):
        self.root = root
        self.mask_labels = ['BG', 'without_mask', 'with_mask', 'mask_weared_incorrect']
        self.imgs = list(sorted(os.listdir('datasets/andrewmvd_FaceMaskDataset/images/')))
        # Add for data augmentation
        bbox_params = A.BboxParams(format = 'pascal_voc', label_fields = ['class_labels'])
        self.transform = A.Compose([ToTensor()], bbox_params = bbox_params)
    

    def __getitem__(self, idx):
        data = 'maksssksksss' + str(idx)

        image_path = f'{self.root}/images/{data}.png'
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        anno_path = f'{self.root}/annotations/{data}.xml'
        parser = parse(anno_path)
        labels, boxes = [], []
        for obj in parser.findall('object'):
            bndbox = obj.find('bndbox') # Add
            if bndbox.find('xmin').text != bndbox.find('xmax').text: # Add
                box = [int(tag.text) for tag in obj.find('bndbox')]
                label = obj.find('name').text
                label = self.mask_labels.index(label)
                boxes.append(box)
                labels.append(label)
                
        transformed = self.transform(image = image, bboxes = boxes, class_labels = labels)
        image = transformed['image']
        boxes = torch.tensor(transformed['bboxes'], dtype = torch.float32)
        labels = torch.tensor(transformed['class_labels'], dtype = torch.int64)
        
        # Add for estimation of mean Average Precisions
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((len(parser.findall('object')),), dtype = torch.int64)

        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id, 'area': area, 'iscrowd': iscrowd}

        return image, target


    def __len__(self):
        return len(self.imgs)
    
pl.seed_everything(0)
root = 'datasets/andrewmvd_FaceMaskDataset/'
dataset1 = MaskDataset(root)

removed_list = []

for i in range(len(dataset1)):
    try:
        x, t = dataset1[i]
    except Exception as e:
        removed_list.append(i)

dataset = []
for i in range(len(dataset1)):
    if i in removed_list:
        pass
    else:
        dataset.append(dataset1[i])

dataset1 = dataset

train1, val1, test1 = torch.utils.data.random_split(dataset = dataset1, lengths = [len(dataset1)-90, 40, 50], generator = torch.Generator().manual_seed(4))

def collate_fn(batch):
    return list(zip(* batch))

batch_size = 8

train_loader = torch.utils.data.DataLoader(train1, batch_size, shuffle = False, drop_last = True, collate_fn = collate_fn)

pl.seed_everything(0)
dataset2 = MaskDatasetNoAug(root)

dataset = []
removed = []
for i in range(len(dataset2)):
    if i in removed_list:
        removed.append(dataset2[i])
    else:
        dataset.append(dataset2[i])

dataset2 = dataset

train2, val2, test2 = torch.utils.data.random_split(dataset = dataset2, lengths = [len(dataset2)-90, 40, 50], generator = torch.Generator().manual_seed(42))

val = []
for i in range(len(val2)):
    val.append(val2[i])
for i in range(len(removed)):
    val.append(removed[i])

val2 = val

val_loader = torch.utils.data.DataLoader(val2, batch_size, collate_fn = collate_fn)
test_loader = torch.utils.data.DataLoader(test2, batch_size, collate_fn = collate_fn)

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
    
num_epochs = 2
pl.seed_everything(0)
net = Net()
trainer = pl.Trainer(max_epochs = num_epochs, deterministic = True)
trainer.fit(net, train_loader, val_loader)

trainer.test(dataloaders = test_loader)

torch.save(net.model, "trained_models/retinanet_"+str(num_epochs)+"epoc")
