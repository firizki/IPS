from mtcnn.mtcnn import MTCNN

class PredictionEngine:
    def __init__(self):
        self.detector_mtcnn = MTCNN()

    def mtcnn(self, image):
        return self.detector_mtcnn.detect_faces(image)
