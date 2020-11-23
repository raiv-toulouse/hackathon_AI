# -*- coding: utf-8 -*-

from PyQt5.QtCore import *
import subprocess
import time


class Thread_Command(QThread):

    display_msg_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super(Thread_Command, self).__init__(parent)

    def exec_command(self,cmd,end_msg):
        self.cmd = cmd
        self.end_msg = end_msg
        self.start()

    def run(self):
        process = subprocess.Popen(self.cmd,stdout=subprocess.PIPE,universal_newlines=True,shell=True)
        while True:
            output = process.stdout.readline()
            self.display_msg_signal.emit(output.strip())
            # Do something else
            return_code = process.poll()
            if return_code is not None:
                # Process has finished, read rest of the output
                for output in process.stdout.readlines():
                    self.display_msg_signal.emit(output.strip())
                break
        self.display_msg_signal.emit(self.end_msg)

