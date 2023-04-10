import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd

from PIL import Image
from PIL import ImageTk
from imagelib import ImageLib

import pickle
import numpy as np
from tensorflow import keras

class IPS:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title('Image Processing Simulator')
        self.window.geometry('800x500')

        self.image_input = tk.Canvas(self.window, width = 400, height = 400)
        self.image_input.grid(row=2,column=1)
        self.image_output = tk.Canvas(self.window, width = 400, height = 400)
        self.image_output.grid(row=2,column=2)

        self.prediction_var = tk.StringVar()
        self.prediction_label = tk.Label(self.window, textvariable=self.prediction_var, relief=tk.RAISED)
        self.prediction_label.grid(row=3,column=1)

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

        test_image=keras.preprocessing.image.load_img(self.filename,target_size=(64, 64))
        test_image=keras.preprocessing.image.img_to_array(test_image)
        test_image=np.expand_dims(test_image,axis=0)
        result=self.classifier.predict(test_image)

        self.prediction_var.set("Prediction is: " + self.ResultMap[np.argmax(result)])

        self.image_input.create_image(20,20, anchor='nw', image=self.image_tk)
        self.image_output.create_image(20,20, anchor='nw', image=self.image_tk)

if __name__ == "__main__":
    ips = IPS()
    ips.window.mainloop()
