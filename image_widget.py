import tkinter

class InputImage(tkinter.Canvas):
    def __init__(self, parent):
        canvas_width=400
        canvas_height=350
        tkinter.Canvas.__init__(self,parent, width = canvas_width, height = canvas_height)
        self.grid(column=0, row=2)
        self.create_text(canvas_width/2, canvas_height-20, text="Image Input", font=("Arial", 20))
        background_color = "white"
        self.configure(background=background_color)

class OutputImage(tkinter.Canvas):
    def __init__(self, parent):
        canvas_width=400
        canvas_height=350
        tkinter.Canvas.__init__(self,parent, width = canvas_width, height = canvas_height)
        self.grid(column=1, row=2)
        self.create_text(canvas_width/2, canvas_height-20, text="Image Output", font=("Arial", 20))
        background_color = "white"
        self.configure(background=background_color)
