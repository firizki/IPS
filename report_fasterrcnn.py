import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

confidence_threshold = 0.9
iou_threshold = 0.9
line_pair =[
    (0.6,0.7),(0.6,0.8),(0.6,0.9),
    (0.7,0.7),(0.7,0.8),(0.7,0.9),
    (0.8,0.7),(0.8,0.8),(0.8,0.9),
    (0.9,0.7),(0.9,0.8),(0.9,0.9)]

df = pd.read_csv("metrics_ssd.csv")

filtered_data = [df.loc[(df['iou_threshold'] == lp[0]) & (df['confidence_threshold'] == lp[1])] for lp in line_pair]

for filtered_df in filtered_data:
    label=f"Threshold  Confidence={filtered_df.iloc[0]['confidence_threshold']} IOU={filtered_df.iloc[0]['iou_threshold']}"
    sns.lineplot(x='total_epoch', y='precision', data=filtered_df, label=label)

plt.xlabel('Total Epoch')
plt.ylabel('Precision')
plt.xticks(range(10, 101, 10))
plt.show()

for filtered_df in filtered_data:
    label=f"Threshold  Confidence={filtered_df.iloc[0]['confidence_threshold']} IOU={filtered_df.iloc[0]['iou_threshold']}"
    sns.lineplot(x='total_epoch', y='recall', data=filtered_df, label=label)

plt.xlabel('Total Epoch')
plt.ylabel('Recall')
plt.xticks(range(10, 101, 10))
plt.show()

for filtered_df in filtered_data:
    label=f"Threshold  Confidence={filtered_df.iloc[0]['confidence_threshold']} IOU={filtered_df.iloc[0]['iou_threshold']}"
    sns.lineplot(x='total_epoch', y='accuracy', data=filtered_df, label=label)

plt.xlabel('Total Epoch')
plt.ylabel('Accuracy')
plt.xticks(range(10, 101, 10))
plt.show()

for filtered_df in filtered_data:
    label=f"Threshold  Confidence={filtered_df.iloc[0]['confidence_threshold']} IOU={filtered_df.iloc[0]['iou_threshold']}"
    sns.lineplot(x='total_epoch', y='f1', data=filtered_df, label=label)

plt.xlabel('Total Epoch')
plt.ylabel('F1-Score')
plt.xticks(range(10, 101, 10))
plt.show()