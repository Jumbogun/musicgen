import numpy as np
import matplotlib.pyplot as plt
import pypianoroll as pr
import pretty_midi
import os
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.models as models  
import torchvision.transforms as transforms
from torch.nn.utils import spectral_norm
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import Parameter
import shutil
import time
import qtawesome
import sys

import pyaudio
import wave
from pyaudio import PyAudio
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.Qt import QMutex
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FC
from util import *
from interface import *
from model import *
from savewindow import *

class AudioThread(QThread):  # 线程1
    def __init__(self, box):
        super().__init__()
        self.box = box
        self.stop_flag = False
        self.p = PyAudio()
        self.mut = QMutex()
        self.stream = self.p.open(format=pyaudio.paFloat32,
                             channels=1,
                             rate=44100,
                             frames_per_buffer=1024,
                             output=True,)
    def finished(self, *args, **kwargs):
       
        self.stream.close()
        super().finshed(*args, **kwargs)     
    def run(self):
        self.mut.lock() 
        for i in range(self.cur_pixel, self.length):
            if(self.stop_flag):
                self.cur_pixel = i
                break
            #print("audio:  ", i)
            self.stream.write(self.waves[i])
            if(not self.box.canvas_thread.isRunning()):
                self.box.canvas_thread.start()
            else:
                self.box.canvas_thread.quit()
        cur_pixel = 0
        self.mut.unlock()
    def set_waves(self, waves):
        self.stop_flag = True
        self.mut.lock()
        self.waves = waves
        self.length = len(self.waves)
        self.mut.unlock()
        self.stop_flag = False
    def set_cur_pixel(self, cur_pixel):
        self.stop_flag = True
        self.mut.lock()
        print("audio set cur")
        self.cur_pixel = cur_pixel 
        print("audio done set cur", self.cur_pixel)
        self.mut.unlock()
        self.stop_flag = False
    def stop(self):  
        self.stop_flag = True
        print('th drop')
        


