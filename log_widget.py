import tkinter
import logging

class LogWidget(tkinter.Text):
    def __init__(self, parent):
        tkinter.Text.__init__(self,parent,state="disabled")
        self.grid(column=1, row=3)

class LogStreamHandler(logging.StreamHandler):
    def __init__(self, textctrl):
        logging.StreamHandler.__init__(self) # initialize parent
        self.textctrl = textctrl

    def emit(self, record):
        msg = self.format(record)
        self.textctrl.config(state="normal")
        self.textctrl.insert("end", msg + "\n")
        self.flush()
        self.textctrl.config(state="disabled")