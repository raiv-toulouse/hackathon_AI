# -*- coding: utf-8 -*-

from PyQt5.QtCore import *
import cv2
from numpy import *
from time import sleep

def gstreamer_pipeline(
    capture_width=320,
    capture_height=240,
    display_width=320,
    display_height=240,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


class Thread_Camera(QThread):

    # Signaux de communication avec l'IHM
    signalAfficherImage = pyqtSignal(ndarray) # Pour afficher une image dans le QLabel nommé video_frame

    def __init__(self, parent=None):
        super(Thread_Camera, self).__init__(parent)
        self.sourceVideo = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        if not self.sourceVideo.isOpened():
            print("Can't open the camera")
            exit(1)
        self.ind_image = 0
        self.path_image = None
        self.record_images = False

    def start_recording(self,ind,path):
        self.ind_image = ind
        self.path_image = path
        self.record_images = True

    def stop_recording(self):
        self.record_images = False

    def changeCamera(self,ind_camera):
        try:
            self.sourceVideo = cv2.VideoCapture(ind_camera)
        except:
            print('pb when changing camera')

    def run(self):
        """
        Thread main method
        """
        while True:
            ret, img = self.sourceVideo.read()
            if ret:
                self.signalAfficherImage.emit(img)
                if self.record_images:
                    img_path = self.path_image/(str(self.ind_image)+'.jpg')
                    cv2.imwrite(str(img_path),img)
                    self.ind_image += 1
                    sleep(0.3)

