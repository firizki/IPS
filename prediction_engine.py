import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO
from PIL import Image

class PredictionEngine:
    def __init__(self):
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.confidence_threshold = 0.8
        
        # Faster RCNN #
        self.model_fasterrcnn = self.get_model_instance_segmentation(3)
        self.model_fasterrcnn.to(torch.device('cpu'))
        checkpoint = torch.load("trained_models/fasterrcnn/checkpoint_99.pth")
        self.model_fasterrcnn.load_state_dict(checkpoint['model_state_dict'])
        self.model_fasterrcnn.eval()

        # RetinaNet #
        self.model_retinanet = torchvision.models.detection.retinanet_resnet50_fpn(weights=None, num_classes = 3)
        self.model_retinanet.to(torch.device('cpu'))
        checkpoint = torch.load("trained_models/retinanet/checkpoint_retinanet_99.pth")
        self.model_retinanet.load_state_dict(checkpoint['model_state_dict'])
        self.model_retinanet.eval()

        # SSD #
        self.model_ssd = torchvision.models.detection.ssd300_vgg16(weights=None, num_classes = 3)
        self.model_ssd.to(torch.device('cpu'))
        checkpoint = torch.load("trained_models/ssd/checkpoint_ssd_99.pth")
        self.model_ssd.load_state_dict(checkpoint['model_state_dict'])
        self.model_ssd.eval()

        # YOLOv8 #
        self.model_yolov8 = YOLO(model="trained_models/yolov8/train100epochs/weights/best.pt")

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
    
    def SSD(self, img_path):
        image_input = Image.open(img_path).convert('RGB')
        tensor_image = self.image_transform(image_input)
        preds = self.model_ssd([tensor_image])

        boxes = preds[0]['boxes'].cpu().detach().numpy()
        labels = preds[0]['labels'].cpu().detach().numpy()
        scores = preds[0]['scores'].cpu().detach().numpy()

        results = [] 
        for box, label, score in zip(boxes, labels, scores):
            if score < 0.5:
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