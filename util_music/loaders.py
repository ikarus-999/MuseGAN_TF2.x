import pickle
import os

import pandas as pd
from PIL import Image
import numpy as np
from os import walk, getcwd
import h5py

import imageio
from glob import glob

from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import pdb

import pypianoroll
import argparse
import os
import os.path as osp
from tqdm import tqdm


# +
def load_music(data_name, filename, n_bars, n_steps_per_bar):
    file = os.path.join("./data", data_name, filename)

    with np.load(file, encoding='bytes', allow_pickle = True) as f:
        data = f['train']

    data_ints = []

    for x in data:
        counter = 0
        cont = True
        while cont:
            if not np.any(np.isnan(x[counter:(counter+4)])):
                cont = False
            else:
                counter += 4

        if n_bars * n_steps_per_bar < x.shape[0]:
            data_ints.append(x[counter:(counter + (n_bars * n_steps_per_bar)),:])


    data_ints = np.array(data_ints)

    n_songs = data_ints.shape[0]
    n_tracks = data_ints.shape[2]

    data_ints = data_ints.reshape([n_songs, n_bars, n_steps_per_bar, n_tracks])

    max_note = 83

    where_are_NaNs = np.isnan(data_ints)
    data_ints[where_are_NaNs] = max_note + 1
    max_note = max_note + 1

    data_ints = data_ints.astype(int)

    num_classes = max_note + 1

    data_binary = np.eye(num_classes)[data_ints]
    data_binary[data_binary==0] = -1
    data_binary = np.delete(data_binary, max_note, -1)

    data_binary = data_binary.transpose([0, 1, 2, 4, 3])
    return data_binary, data_ints, data

def process_dataset(in_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    in_filename_list, out_filename_list = [], []
    for main_dir, sub_dir, filename_list in os.walk(in_dir):
        for filename in filename_list:
            if '.npz' not in filename:
                continue
            in_filename_list.append(osp.join(main_dir, filename))
            track_name = main_dir.split("/")[-1]
            out_filename_list.append(osp.join(out_dir, track_name + '.mid'))

    for i, in_filename in enumerate(tqdm(in_filename_list)):
        convert_midi(in_filename, out_filename_list[i])


# -

import pypianoroll
import argparse
import os
import os.path as osp
from tqdm import tqdm


# +
def convert_midi(in_filename, out_filename):
    pianoroll = pypianoroll.load(in_filename)
    pypianoroll.write(out_filename, pianoroll)

def process_dataset(in_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    in_filename_list, out_filename_list = [], []
    for main_dir, sub_dir, filename_list in os.walk(in_dir):
        for filename in filename_list:
            if '.npz' not in filename:
                continue
            in_filename_list.append(osp.join(main_dir, filename))
            track_name = main_dir.split("/")[-1]
            out_filename_list.append(osp.join(out_dir, track_name + '.mid'))

    for i, in_filename in enumerate(tqdm(in_filename_list)):
        convert_midi(in_filename, out_filename_list[i])
