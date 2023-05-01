import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd

from PIL import Image
from PIL import ImageTk
from imagelib import ImageLib

import pickle
import logging
import datetime
import numpy as np
from tensorflow import keras

from log_widget import LogWidget, LogStreamHandler
from image_widget import InputImage, OutputImage

module_logger = logging.getLogger(__name__)

class IPS:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title('Image Processing Simulator')
        self.window.geometry('800x500')

        self.image_input = InputImage(self.window)
        self.image_output = OutputImage(self.window)

        self.log_window = LogWidget(self.window)

        self.classifier = keras.models.load_model('models/face-test-1')
        file = open('models/face-test-1/ResultsMap.pkl', 'rb')
        self.ResultMap = pickle.load(file)
        file.close()

        open_button = ttk.Button(
            self.window,
            text='Open a File',
            command=self.select_file
        )

        open_button.grid(sticky="W",row=1,column=1)

    def select_file(self):
        filetypes = (
            ('Image files', '.png .jpg .jpeg'),
            ('All files', '*.*')
        )

        self.filename = fd.askopenfilename(filetypes=filetypes)
        self.image_data = Image.open(self.filename)
        self.image_tk = ImageTk.PhotoImage(self.image_data)

        module_logger.info("Load image input")

        test_image=keras.preprocessing.image.load_img(self.filename,target_size=(64, 64))
        test_image=keras.preprocessing.image.img_to_array(test_image)
        test_image=np.expand_dims(test_image,axis=0)
        result=self.classifier.predict(test_image)

        module_logger.info("Prediction is: " + self.ResultMap[np.argmax(result)])

        self.image_input.create_image(20,20, anchor='nw', image=self.image_tk)
        self.image_output.create_image(20,20, anchor='nw', image=self.image_tk)

if __name__ == "__main__":
    ips = IPS()

    stderrHandler = logging.StreamHandler()
    module_logger.addHandler(stderrHandler)
    guiHandler = LogStreamHandler(ips.log_window)
    module_logger.addHandler(guiHandler)
    module_logger.setLevel(logging.INFO)
    now = datetime.datetime.now()
    module_logger.info(str(now) + " Starting application...")  

    ips.window.mainloop()
