import tkinter
import logging

class LogWidget(tkinter.Text):
    def __init__(self, parent):
        self.scrollbar = tkinter.Scrollbar(parent)
        self.scrollbar.pack(side="right", fill="y")

        tkinter.Text.__init__(self, parent, yscrollcommand=self.scrollbar.set)
        self.pack(side="left", fill="both", expand=True)

        self.scrollbar.config(command=self.yview)


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