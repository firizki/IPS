import pickle
import numpy as np

from mtcnn.mtcnn import MTCNN
from tensorflow import keras
from matplotlib import pyplot

class PredictionEngine:
    def __init__(self):
        self.detector_mtcnn = MTCNN()

        self.classifier = keras.models.load_model('models/face-test-1')
        file = open('models/face-test-1/ResultsMap.pkl', 'rb')
        self.ResultMap = pickle.load(file)
        file.close()

    def mtcnn(self, image):
        pixels = pyplot.imread(image)
        return self.detector_mtcnn.detect_faces(pixels)

    def keras(self, image):
        test_image=keras.preprocessing.image.load_img(image,target_size=(64, 64))
        test_image=keras.preprocessing.image.img_to_array(test_image)
        test_image=np.expand_dims(test_image,axis=0)
        return self.classifier.predict(test_image)