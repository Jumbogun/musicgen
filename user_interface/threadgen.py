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


class GenThread(QThread):  # 线程1
    donesignal =pyqtSignal()
    def __init__(self, args, gen_model, pixelcnn, device):
        super().__init__()
        self.gen_model = gen_model.eval()
        self.pixelcnn = pixelcnn.eval()
        self.device = device
        
        self.program_nums     = args["program_nums"]
        self.is_drum          = args["is_drum"]
        self.tempo            = args["tempo"]
        self.n_tracks         = args["n_tracks"]
        self.n_pitches        = args["n_pitches"]
        self.n_bars         = args["n_bars"]
        self.lowest_pitch     = args["lowest_pitch"]
        self.n_measures       = args["n_measures"]
        self.beat_resolution  = args["beat_resolution"]
        self.batch_size       = args["batch_size"]
        self.decay            = args["decay"]
        self.commitment_cost  = args["commitment_cost"]
        self.latent_w         = args["latent_w"]
        self.latent_h         = args["latent_h"]
        self.latent_dim       = args["latent_dim"]
        self.embedding_dim    = args["embedding_dim"]
        self.num_embeddings   = args["num_embeddings"]
        self.scalt_factor = 24 // self.beat_resolution   

        self.resolution = 4 * args["n_measures"] * args["beat_resolution"]  #64 # bar resolution
        
        self.mut = QMutex()
    def generate(self, n_bars, tracks_selection):
    
        self.mut.lock()
        self.n_bars = n_bars
        self.tracks_selection = tracks_selection
        self.start()
    def run(self):
        self.pr = self.generate_pianroll()
        print("done gen")
        self.donesignal.emit()
        self.mut.unlock()
        
    def get_output(self):
        self.mut.lock()
        self.mut.unlock()
        return self.pr  
        
    def generate_pianroll(self):
        batch_size = 1
        Z_track, Z_w, Z_h = self.n_tracks, self.n_bars * self.latent_w, self.latent_h
        rand_idx = torch.autograd.Variable(torch.multinomial(torch.rand(batch_size * Z_track * Z_w * Z_h, self.embedding_dim),1)).squeeze().long().cuda()
        rand_Z = self.gen_model.VQ._embedding.weight[rand_idx].view(-1, Z_track, Z_h, Z_w, self.embedding_dim).permute(0, 4, 1, 2, 3)
        starting_point = 0
        for i in range(Z_w):
                if(i < starting_point):
                    continue
                with torch.no_grad():
                    logit = self.pixelcnn(rand_Z)
                prob = F.softmax(logit, dim = 1).data
                prob = prob[:,:,:,:, i].permute(0, 2, 3, 1).reshape(-1, self.num_embeddings)
                idx = torch.multinomial(prob,1).reshape(logit.shape[0], logit.shape[2],logit.shape[3])
                #idx = prob[:,:,4,4].argmax(dim = 1).squeeze()
                
                for t in range(self.n_tracks):
                    if(not self.tracks_selection[t]):
                        idx[:, t] = 3
                
                rand_Z[:,:,:,:,i] = self.gen_model.VQ._embedding.weight[idx].reshape(logit.shape[0], logit.shape[2],logit.shape[3], self.embedding_dim).permute(0, 3, 1, 2)

        x = rand_Z.reshape(batch_size, self.embedding_dim, self.n_tracks, self.latent_h, self.n_bars, self.latent_w)
        x = x.permute(0, 4, 2, 1, 5, 3).contiguous()
        gen_bar = self.gen_model.decoder(x[0])

        pr = gen_bar.cpu().detach().numpy().reshape(self.n_bars, self.n_tracks, self.resolution, self.n_pitches).transpose(0, 2, 3, 1)
        pr = np.stack([pr] * self.scalt_factor, axis = 2)
        pr = pr.reshape(pr.shape[0], -1, pr.shape[3], pr.shape[4])

        
        return pr

        
      




