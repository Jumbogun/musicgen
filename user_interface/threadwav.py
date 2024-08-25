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
import math
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
from model import *
class WavThread(QThread):  # 
    donesignal =pyqtSignal()
    def __init__(self):
        super().__init__()
        self.mut = QMutex()
        
    def generate(self, pr, length, program_nums, is_drum, tempo):
        self.mut.lock()
        self.pr = pr
        self.length = length
        self.program_nums = program_nums
        self.is_drum = is_drum
        self.tempo = tempo
        self.length = length
        self.start()
        
    def run(self):
        self.waves, self.pm, self.wav = self.get_wav() 
        self.donesignal.emit()  
        self.mut.unlock()
             
    def get_wav(self):
        pm = get_midis(self.pr  > 0.5, self.program_nums, self.is_drum, self.tempo)
        midi_audio = pm.fluidsynth(fs = 44100, sf2_path="./soundfont/TimGM6mb.sf2")
        midi_audio = midi_audio.astype(np.float32)
        wav_len = math.ceil(midi_audio.shape[0] / self.length) * self.length
        n_pad = wav_len - midi_audio.shape[0] 
        wav = np.concatenate([midi_audio, np.zeros([n_pad], np.float32)], axis = 0).reshape(self.length, -1)
        midi_audio = wav.reshape(self.length, -1)
        
        waves = []
        for i in range(self.length):
            waves.append(midi_audio[i].tobytes())
        return waves, pm, wav
        
    def get_output(self):
        self.mut.lock()
        self.mut.unlock()
        return self.waves, self.pm, self.wav
