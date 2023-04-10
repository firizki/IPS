# Deep Learning CNN model to recognize face
'''This script uses a database of images and creates CNN model on top of it to test
   if the given image is recognized correctly or not'''

'''####### IMAGE PRE-PROCESSING for TRAINING and TESTING data #######'''

# Specifying the folder where images are present

import pickle
import numpy as np
from tensorflow import keras

classifier = keras.models.load_model('models/face-test-1')
 
ImagePath='datasets/Face-Images/Face Images/Final Testing Images/face6/1face6.jpg'
test_image=keras.preprocessing.image.load_img(ImagePath,target_size=(64, 64))
test_image=keras.preprocessing.image.img_to_array(test_image)
 
test_image=np.expand_dims(test_image,axis=0)
 
result=classifier.predict(test_image)
#print(training_set.class_indices)

file = open('models/face-test-1/ResultsMap.pkl', 'rb')

ResultMap = pickle.load(file)

file.close()

print('####'*10)
print('Prediction is: ',ResultMap[np.argmax(result)])