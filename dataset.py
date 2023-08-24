import os
import numpy as np
import obspy
import torch
import random
from SNR import gen_gauss_noise
from torch.utils.data import Dataset
from config_param import Config
from sklearn import preprocessing   # introducing Z-score standardization
from Max_Min_Normalize import max_min_normalization

class My_Dataset(Dataset):
    def __init__(self, train_files, train_labels, mode='train'):
        self.train_files = train_files
        self.train_labels = train_labels
        self.mode = mode
    
    def __len__(self):
        return len(self.train_files)
    
    def __getitem__(self, idx):
        data_dir = Config().data_dir
        label_dir = Config().label_dir
        data_path = os.path.join(data_dir, self.train_files[idx])
        label_path = os.path.join(label_dir, self.train_labels[idx])
        
        data = np.zeros((3, 20000))
        data[0, :] = obspy.read(data_path + '/' + self.train_files[idx] + '.EHE.sac')[0].data
        data[1, :] = obspy.read(data_path + '/' + self.train_files[idx] + '.EHN.sac')[0].data
        data[2, :] = obspy.read(data_path + '/' + self.train_files[idx] + '.EHZ.sac')[0].data
        # Add noise
        for i in range(3):
            SNR = random.uniform(1, 10) # set the SNR to a random value
            data[i, :] = data[i, :] + gen_gauss_noise(data[i, :], SNR)
        
        label = np.zeros((3, 20000))
        label[0, :] = obspy.read(label_path + '/' + self.train_files[idx] + '.EHE.sac')[0].data
        label[1, :] = obspy.read(label_path + '/' + self.train_files[idx] + '.EHN.sac')[0].data
        label[2, :] = obspy.read(label_path + '/' + self.train_files[idx] + '.EHZ.sac')[0].data
        
        for i in range(3):
            data[i, :] = max_min_normalization(data[i, :])
            label[i, :] = max_min_normalization(label[i, :])
            # data[i, :] = preprocessing.scale(data[i, :])
            # label[i, :] = preprocessing.scale(label[i, :])
        return torch.from_numpy(data.astype(np.float32)), torch.from_numpy(label.astype(np.float32))    