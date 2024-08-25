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

class CanvasThread(QThread):  # 
    def __init__(self, box):
        super().__init__()
        self.box = box
    def run(self):
        self.box.inc_cur_pixel()
