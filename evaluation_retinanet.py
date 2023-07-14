import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

class_to_int = {'background': 0, 'face_with_mask': 1, 'face_no_mask': 2}
int_to_class = {0: 'background', 1: 'face_with_mask', 2: 'face_no_mask'}

# def plot_img(image_name, plot_results):
    
#     fig, ax = plt.subplots(1, 2, figsize = (14, 14))
#     ax = ax.flatten()
    
#     bbox = df_ground[df_ground['name'] == image_name]
#     img_path = os.path.join("datasets/merged_dataset/images/", image_name)
    
#     image = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
#     image /= 255.0
#     image2 = image.copy()
    
#     for idx, row in bbox.iterrows():
#         x1 = row['x1']
#         y1 = row['y1']
#         x2 = row['x2']
#         y2 = row['y2']
#         label = row['classname']
        
#         cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 3)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(image, label, (int(x1),int(y1-10)), font, 1, (255,0,0), 2)
    
#     ax[0].set_title('Original Image')
#     ax[0].imshow(image)

#     for pr in plot_results:
#         x1 = pr[1]
#         y1 = pr[2]
#         x2 = pr[3]
#         y2 = pr[4]
#         label = str(pr[6])
#         cv2.rectangle(image2, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 3)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(image2, label, (int(x1),int(y1-10)), font, 1, (255,0,0), 2)

#     ax[1].set_title('Image with Bondary Box')
#     ax[1].imshow(image2)

#     plt.show()

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    return iou

file = open("valid_dataset.txt", "r")
test_files = []
for line in file:
    test_files.append(line.strip())
file.close()
df_ground = pd.read_csv("datasets/merged_dataset/label.csv")

metrics_file = open("metrics_retinanet.csv", "a")

confidence_thresholds = [0.7,0.8,0.9]
iou_thresholds = [0.6,0.7,0.8,0.9]
epochs = [9,19,29,39,49,59,69,79,89,99]

for epoch in tqdm(epochs):
    for confidence_threshold in confidence_thresholds:
        for iou_threshold in iou_thresholds:

            df_result = pd.read_csv(f"result_test/result_test_retinanet_{epoch}.csv")

            total_tp = 0
            total_fp = 0
            total_fn = 0

            for index_target_name in range(len(test_files)):
                target_name = test_files[index_target_name]

                source_grounds = df_ground[df_ground["name"] == target_name].values
                pred_results = df_result[df_result["name"] == target_name].values

                newtmp = []
                for i in range(len(pred_results)):
                    if pred_results[i][6] >= confidence_threshold:
                        newtmp.append(pred_results[i])

                pred_results = newtmp

                removed_grounds = []
                removed_results = []
                tp=0
                fp=0

                for i in range(len(source_grounds)):
                    if i in removed_grounds:
                        continue
                    for j in range(len(pred_results)):
                        if j in removed_results:
                            continue

                        box_ground = [source_grounds[i][1],source_grounds[i][2],source_grounds[i][3],source_grounds[i][4]]
                        box_pred = [pred_results[j][1],pred_results[j][2],pred_results[j][3],pred_results[j][4]]

                        iou = calculate_iou(box_ground, box_pred)
                        if iou > iou_threshold:
                            if source_grounds[i][5]==int_to_class[pred_results[j][5]]:
                                tp+=1
                            else:
                                fp+=1
                            removed_grounds.append(i)
                            removed_results.append(j)
                            break

                total_tp += tp
                total_fp += fp
                total_fn += (len(source_grounds)-len(removed_grounds))+(len(pred_results)-len(removed_results))

            total_epoch=epoch+1
            precision=(total_tp)/(total_tp+total_fp)
            recall=(total_tp)/(total_tp+total_fn)
            accuracy=(total_tp)/(total_tp+total_fp+total_fn)
            f1=(2*total_tp)/((2*total_tp)+total_fp+total_fn)

            metrics_file.write(f"{total_epoch},{confidence_threshold},{iou_threshold},{precision},{recall},{accuracy},{f1}\n")
            # print(f"total_epoch={epoch+1}")
            # print(f"confidence_threshold={confidence_threshold}")
            # print(f"iou_threshold={iou_threshold}")
            # print(f"TP={total_tp} FP={total_fp} FN={total_fn}")
            # print(f"Precision={(total_tp)/(total_tp+total_fp)}")
            # print(f"Recall={(total_tp)/(total_tp+total_fn)}")
            # print(f"Accuracy={(total_tp)/(total_tp+total_fp+total_fn)}")
            # print(f"F1={(2*total_tp)/((2*total_tp)+total_fp+total_fn)}")
