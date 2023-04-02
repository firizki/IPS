import tkinter as tk
from tkinter import filedialog as fd

from PIL import Image
from PIL import ImageTk

class ImageLib:
    def __init__(self, window):
        self.canvas = tk.Canvas(window, width = 400, height = 400)

    def select_file(self):
        filetypes = (
            ('Image files', '.png .jpg .jpeg'),
            ('All files', '*.*')
        )

        self.filename = fd.askopenfilename(filetypes=filetypes)
        self.image_data = Image.open(self.filename)
        self.image_tk = ImageTk.PhotoImage(self.image_data) 
        self.canvas.create_image(20,20, anchor='nw', image=self.image_tk)      
