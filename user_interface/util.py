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

def piano_rolls_to_midi(piano_rolls, program_nums=None, is_drum=None, velocity=100, tempo=120.0, beat_resolution=24):
    if len(piano_rolls) != len(program_nums) or len(piano_rolls) != len(is_drum):
        print( "Error: piano_rolls and program_nums have different sizes...")
        return False
    if not program_nums:
        program_nums = [0, 0, 0]
    if not is_drum:
        is_drum = [False, False, False]
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    # Iterate through all the input instruments
    for idx in range(len(piano_rolls)):
        # Create an Instrument object
        instrument = pretty_midi.Instrument(program=program_nums[idx], is_drum=is_drum[idx])
        # Set the piano roll to the Instrument object
        set_piano_roll_to_instrument(piano_rolls[idx], instrument, velocity, tempo, beat_resolution)
        # Add the instrument to the PrettyMIDI object
        midi.instruments.append(instrument)
    return midi
def set_piano_roll_to_instrument(piano_roll, instrument, velocity=100, tempo=120.0, beat_resolution=24):
    # Calculate time per pixel
    tpp = 60.0/tempo/float(beat_resolution)
    # Create piano_roll_search that captures note onsets and offsets
    piano_roll = piano_roll.reshape((piano_roll.shape[0] * piano_roll.shape[1], piano_roll.shape[2]))
    piano_roll_diff = np.concatenate((np.zeros((1,128),dtype=int), piano_roll, np.zeros((1,128),dtype=int)))  
    piano_roll_search = np.diff(piano_roll_diff.astype(int), axis=0)
    # Iterate through all possible(128) pitches
    for note_num in range(128):
        # Search for notes
        start_idx = (piano_roll_search[:,note_num] > 0).nonzero()
        start_time = tpp*(start_idx[0].astype(float))
        end_idx = (piano_roll_search[:,note_num] < 0).nonzero()
        end_time = tpp*(end_idx[0].astype(float))
        # Iterate through all the searched notes
        for idx in range(len(start_time)):
            # Create an Note object with corresponding note number, start time and end time
            note = pretty_midi.Note(velocity=velocity, pitch=note_num, start=start_time[idx], end=end_time[idx])
            # Add the note to the Instrument object
            instrument.notes.append(note)
    # Sort the notes by their start time
    instrument.notes.sort(key=lambda note: note.start)

#_Bass_, _Drums_, _Guitar_,_Strings_, _Piano_.
#*Drums*, *Piano*, *Guitar*, *Bass*, *Ensemble*, *Reed*, *Synth Lead* and *Synth Pad*.
def get_midis(bars, program_nums, is_drum, tempo):
    padded_bars = np.concatenate((np.zeros((bars.shape[0], bars.shape[1], 24, bars.shape[3])), bars, np.zeros((bars.shape[0], bars.shape[1], 32, bars.shape[3]))), axis=2)

    images_with_pause = padded_bars
    images_with_pause = images_with_pause.reshape(-1, 384, padded_bars.shape[2], padded_bars.shape[3])
    images_with_pause_list = []
    for ch_idx in range(padded_bars.shape[3]):
        images_with_pause_list.append(images_with_pause[:,:,:,ch_idx].reshape(images_with_pause.shape[0],  \
                                                        images_with_pause.shape[1], images_with_pause.shape[2]))
    return piano_rolls_to_midi(images_with_pause_list, program_nums, is_drum,
                               tempo=tempo)
