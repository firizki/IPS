import tkinter

class InputImage(tkinter.Canvas):
    def __init__(self, parent):
        tkinter.Canvas.__init__(self,parent, width = 400, height = 400)
        self.grid(column=1, row=2)

class OutputImage(tkinter.Canvas):
    def __init__(self, parent):
        tkinter.Canvas.__init__(self,parent, width = 400, height = 400)
        self.grid(column=2, row=2)
