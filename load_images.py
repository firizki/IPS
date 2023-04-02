import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf

print(tf.__version__)

import pathlib
data_dir = pathlib.Path('datasets\Face-Images\Face Images\Final Training Images').with_suffix('')
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

batch_size = 4
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  labels='inferred')