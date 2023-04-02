import tkinter as tk
from tkinter import ttk

from imagelib import ImageLib

class IPS:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title('Image Processing Simulator')
        self.window.geometry('500x500')
        image_output = ImageLib(self.window)
        image_output.canvas.grid(row=2,column=1)

        open_button = ttk.Button(
            self.window,
            text='Open a File',
            command=image_output.select_file
        )

        open_button.grid(sticky="W",row=1,column=1)

if __name__ == "__main__":
    ips = IPS()
    ips.window.mainloop()
