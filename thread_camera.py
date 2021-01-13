# -*- coding: utf-8 -*-

from PyQt5.QtCore import QThread,pyqtSignal
import cv2
from numpy import *
from time import sleep
import jetson.utils


class Thread_Camera(QThread):

    # Signaux de communication avec l'IHM
    signalAfficherImage = pyqtSignal(ndarray,int) # Pour afficher une image dans le QLabel nommé video_frame

    def __init__(self, parent=None):
        super(Thread_Camera, self).__init__(parent)
        #self.sourceVideo = jetson.utils.videoSource("csi://0",argv=['threadCamera', '--input-width=320', '--input-height=240', '--input-flip=none'])
        self.sourceVideo = jetson.utils.videoSource("/dev/video0",argv=['threadCamera', '--input-width=320', '--input-height=240', '--input-flip=none'])
        self.ind_image = 0
        self.path_image = None
        self.record_images = False
        self.camera_running = True

    def start_recording(self,ind,path):
        self.ind_image = ind
        self.path_image = path
        self.record_images = True

    def stop_recording(self):
        self.record_images = False

    def run(self):
        """
        Thread main method
        """
        while self.camera_running:
            cuda_img = self.sourceVideo.Capture()
            img = jetson.utils.cudaToNumpy(cuda_img)
            self.signalAfficherImage.emit(img, self.ind_image)
            if self.record_images:
                img_path = self.path_image/(str(self.ind_image)+'.jpg')
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                cv2.imwrite(str(img_path),img)
                self.ind_image += 1
                sleep(0.3)
            sleep(0.1)

