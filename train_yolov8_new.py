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

DIR_INPUT = "datasets/merged_dataset/"
DIR_IMAGES = DIR_INPUT + "images/"
train_path = "datasets/yolodataset/train"
valid_path = "datasets/yolodataset/valid"
test_path = "datasets/yolodataset/test"

class_id = {
    "face_with_mask" : 0,
    "face_no_mask" : 1
}

df_data = pd.read_csv(DIR_INPUT+"label.csv")

# def show_random_images_with_bbox(df):
#     all_images = os.listdir(DIR_IMAGES)
#     random_image_filename = random.sample(all_images, 4)
#     fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
#     for i, filename in enumerate(random_image_filename):
#         print(filename)
#         selected_df = df[df['name'] == filename]
        
#         image = Image.open(DIR_IMAGES + '/' + filename)
        
#         ax.flat[i].imshow(image)
#         ax.flat[i].axis(False)
        
#         image_bboxes = []
#         for df_index in range(0, len(selected_df)):
#             color = "g"
#             if selected_df.iloc[df_index].classname == "face_with_mask": color = "y"
#             elif selected_df.iloc[df_index].classname == "face_no_mask": color = "r"
            
#             x_min = selected_df.iloc[df_index].x1
#             y_min = selected_df.iloc[df_index].y1
#             x_max = selected_df.iloc[df_index].x2
#             y_max = selected_df.iloc[df_index].y2
            
#             rect = patches.Rectangle([x_min, y_min], x_max-x_min, y_max-y_min, 
#                              linewidth=2, edgecolor=color, facecolor="none")
#             ax.flat[i].add_patch(rect)
#     plt.show()
            
# show_random_images_with_bbox(df_data)

train, test = train_test_split(df_data.name.unique(), test_size=0.2, random_state=23)
train, valid = train_test_split(train, test_size=0.15, random_state=23)

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        return width, height

def pascal_voc_to_yolo_bbox(bbox_array, w, h):
    x_min, y_min, x_max, y_max = bbox_array
    
    x_center = ((x_max + x_min) / 2) / w
    y_center = ((y_max + y_min) / 2) / h
    
    width = (x_max - x_min) / w
    height = (y_max - y_min) / h
    
    return [x_center, y_center, width, height]

def copy_image_file(image_items, folder_name):
    for image in image_items:
            image_path = DIR_IMAGES + image
            new_image_path = os.path.join(folder_name, image)
            shutil.copy(image_path, new_image_path)

def create_label_file(image_items, folder_name):
    for image in image_items:
        fileName = Path(image).stem
        df = df_data[df_data['name'] == image]
        with open(folder_name + "/" + fileName +'.txt', 'w') as f:
            for i in range(0, len(df)):
                parse_bbox_array = [df.iloc[i]['x1'], df.iloc[i]['y1'], df.iloc[i]['x2'], df.iloc[i]['y2']]
                parse_width, parse_height = get_image_dimensions(DIR_IMAGES + image)
                bbox = pascal_voc_to_yolo_bbox(parse_bbox_array, parse_width, parse_height)
                bbox_text = " ".join(map(str, bbox))
                txt = str(class_id[df.iloc[i]['classname']])+ " " + bbox_text
                f.write(txt)
                if i != len(df) - 1:
                    f.write("\n")
                
copy_image_file(train, train_path)
copy_image_file(valid, valid_path)
copy_image_file(test, test_path)

create_label_file(train, train_path)
create_label_file(valid, valid_path)
create_label_file(test, test_path)

def walk_through_dir(filepath):
    for dirpath, dirnames, filenames in os.walk(filepath):
        print(f"There are {len(dirnames)} directories and {len(glob.glob(filepath + '/*.png', recursive = True))} images in '{dirpath}'.")
    
walk_through_dir(train_path)
walk_through_dir(valid_path)  
walk_through_dir(test_path)

classes = list(df_data.classname.unique())
class_count = len(classes)
facemask_yaml = f"""
    train: train
    val: valid
    test: test
    nc: {class_count}
    names:
        0 : face_with_mask
        1 : face_no_mask
    """

with open('facemask.yaml', 'w') as f:
    f.write(facemask_yaml)
    
model = YOLO("yolov8n.pt") 
model.train(data="facemask.yaml", epochs=50)