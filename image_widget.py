import tkinter

class InputImage(tkinter.Canvas):
    def __init__(self, parent):
        canvas_width=400
        canvas_height=350
        tkinter.Canvas.__init__(self,parent, width = canvas_width, height = canvas_height)
        self.grid(column=0, row=1)
        background_color = "white"
        self.configure(background=background_color)
        self.label = tkinter.Label(parent, text="Image Input", font=("Arial", 16))
        self.label.grid(column=0, row=0)

class OutputImage(tkinter.Canvas):
    def __init__(self, parent):
        canvas_width=400
        canvas_height=350
        tkinter.Canvas.__init__(self,parent, width = canvas_width, height = canvas_height)
        self.grid(column=1, row=1)
        background_color = "white"
        self.configure(background=background_color)
        self.label = tkinter.Label(parent, text="Image Output", font=("Arial", 16))
        self.label.grid(column=1, row=0)
