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
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from util import *
from interface import *
from model import *

track_code = [{}, {}]
fo = open("instrument_name/drum_name", "r+")

while(True):
    line = fo.readline()
    if(len(line) < 1):
        break
    idx = int(line.split()[0])
    name = line[:-1]
    track_code[0][idx] = name
fo.close()

fo = open("instrument_name/ins_name", "r+")
while(True):
    line = fo.readline()
    if(len(line) < 1):
        break
    idx = int(line.split()[0])
    name = line[:-1]
    track_code[1][idx] = name
fo.close()
args = {
    "program_nums"   :   [24, 0, 25, 33, 49],
    "is_drum"        :   [True, False, False, False, False],
    "tempo"          :   160,
    "n_tracks"       :    5,
    "n_pitches"      :   72,
    "n_bars"       :   12,
    "lowest_pitch"   :   24,
    "n_measures"     :    4,
    "beat_resolution":    4,
    "batch_size"     :    1,
    "decay"          :    0.99,
    "commitment_cost":    0.25,
    "latent_w"       :    4,
    "latent_h"       :    6,
    "latent_dim"     :    8,
    "embedding_dim"  :    8,
    "num_embeddings" : 2048,
}
args["track_code"] = track_code 

device = torch.device("cuda:0" if (torch.cuda.is_available() ) else "cpu")
#load generation model
gen_model = Model(args["commitment_cost"], args["decay"]).to(device)
check_path = './model/musevqvae.pth'
checkpoint = torch.load(check_path)
gen_model.load_state_dict(checkpoint['model'])

pixelcnn = PIXELCNN(k_dim=args["num_embeddings"],z_dim=args["embedding_dim"]).to(device)
check_path = './model/tripixelcnn.pth'
checkpoint = torch.load(check_path)
pixelcnn.load_state_dict(checkpoint['model'])
app=QApplication(sys.argv)
#demo=Box(args, 0, 0, 0)
demo=Box(args, gen_model, pixelcnn, device)
demo.show()
sys.exit(app.exec_())
