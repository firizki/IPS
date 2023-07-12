import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd

from PIL import Image,ImageTk,ImageDraw

import logging
import datetime
import numpy as np

from log_widget import LogWidget, LogStreamHandler
from image_widget import InputImage, OutputImage
from prediction_engine import PredictionEngine

module_logger = logging.getLogger(__name__)

class IPS:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Image Processing Simulator')
        self.root.geometry("800x500")

        self.upload_frame = tk.Frame(self.root)
        self.upload_frame.pack()

        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack()
        
        self.log_frame = tk.Frame(self.root)
        self.log_frame.pack()

        self.image_input = InputImage(self.image_frame)
        self.image_output = OutputImage(self.image_frame)

        self.log_window = LogWidget(self.log_frame)

        self.prediction = PredictionEngine()

        open_button = ttk.Button(
            self.upload_frame,
            text='Choose an image',
            command=self.select_file
        )

        open_button.pack()

    def select_file(self):
        filetypes = (
            ('Image files', '.png .jpg .jpeg'),
            ('All files', '*.*')
        )

        self.filename = fd.askopenfilename(filetypes=filetypes)

        module_logger.info("Load image input: "+ self.filename)

        # Prediction using FasterRCNN
        prediction_result = self.prediction.FasterRCNN(self.filename)
        module_logger.info("Total face(s) found: " + str(len(prediction_result)))

        # Prediction using RetinaNet
        # prediction_result = self.prediction.RetinaNet(self.filename)
        # module_logger.info("Total face(s) found: " + str(len(prediction_result)))

        image_data = Image.open(self.filename)
        image_output = image_data.copy()

        draw = ImageDraw.Draw(image_output)
        for result in prediction_result:
            module_logger.info("Prediction result: " + str(result['confidence']))
            
            x, y, width, height = result['box']
            draw.rectangle((x, y, x+width, y+height), outline="red", width=2)

            # for key in result['keypoints']:
            #     x, y = result['keypoints'][key]
            #     dot_radius = 3
            #     draw.ellipse((x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius), fill="red")

            # dot_radius = 3
            # x, y = result['keypoints']['left_eye']
            # draw.ellipse((x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius), fill="red")
            # x, y = result['keypoints']['right_eye']
            # draw.ellipse((x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius), fill="red")

        image_data.thumbnail((400,350), resample=Image.Resampling.BICUBIC, reducing_gap=2.0)
        image_output.thumbnail((400,350), resample=Image.Resampling.BICUBIC, reducing_gap=2.0)

        self.imagetk_input = ImageTk.PhotoImage(image_data)
        self.imagetk_output = ImageTk.PhotoImage(image_output)
        self.image_input.create_image(0,0, anchor='nw', image=self.imagetk_input)
        self.image_output.create_image(0,0, anchor='nw', image=self.imagetk_output)

if __name__ == "__main__":
    ips = IPS()

    stderrHandler = logging.StreamHandler()
    module_logger.addHandler(stderrHandler)
    guiHandler = LogStreamHandler(ips.log_window)
    module_logger.addHandler(guiHandler)
    module_logger.setLevel(logging.INFO)
    now = datetime.datetime.now()
    module_logger.info(str(now) + " Starting application...")

    ips.root.mainloop()
