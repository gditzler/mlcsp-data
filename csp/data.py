# MIT License
#
# Copyright (c) 2024 Gregory Ditzler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re
import glob 
import numpy as np 
import pandas as pd 

import torch
from torch.utils.data import Dataset, DataLoader

class CSPDataset(Dataset):
    
    def __init__(self, pth_metadata:str, pth_data:list):
        col_names = [
            'index_0', 'index_1', 'index_2', 'symbol_rate', 'carrier_freq_offset', 
            'mod_variant', 'mod_type', 'signal_power'
        ] 
        self.pth_metadata = pth_metadata
        self.pth_data = pth_data
        self.n_samples = 262144
        self.metadata = pd.read_csv(self.pth_metadata, header=None, names=col_names)
        # set self.data 
        self._build_files()
        # set self.labels 
        self._set_labels()
        
    def __len__(self):
        return self.n_files
    
    def _set_labels(self): 
        self.label_name = []
        self.labels = np.zeros((self.n_files,), dtype=np.int16)
        
        for n, file_index in enumerate(self.file_index): 
            slice = self.metadata[self.metadata['index_1'] == file_index]
            if len(slice) == 1: 
                if slice['mod_type'].values[0] == 1 and slice['mod_variant'].values[0] == 1: 
                    self.label_name += ['BPSK']
                    self.labels[n] = 0
                elif slice['mod_type'].values[0] == 1 and slice['mod_variant'].values[0] == 2:
                    self.label_name += ['QPSK']
                    self.labels[n] = 1
                elif slice['mod_type'].values[0] == 1 and slice['mod_variant'].values[0] == 3:  
                    self.label_name += ['8PSK']
                    self.labels[n] = 2
                elif slice['mod_type'].values[0] == 2 and slice['mod_variant'].values[0] == 2:
                    self.label_name += ['4QAM']
                    self.labels[n] = 3
                elif slice['mod_type'].values[0] == 2 and slice['mod_variant'].values[0] == 4:
                    self.label_name += ['16QAM']
                    self.labels[n] = 4
                elif slice['mod_type'].values[0] == 2 and slice['mod_variant'].values[0] == 6:
                    self.label_name += ['64QAM']
                    self.labels[n] = 5
                elif slice['mod_type'].values[0] == 3 and slice['mod_variant'].values[0] == 1:
                    self.label_name += ['SQPSK']
                    self.labels[n] = 6
                elif slice['mod_type'].values[0] == 3 and slice['mod_variant'].values[0] == 2:
                    self.label_name += ['MSK']
                    self.labels[n] = 7
                elif slice['mod_type'].values[0] == 3 and slice['mod_variant'].values[0] == 3:
                    self.label_name += ['GMSK']
                    self.labels[n] = 8
            else: 
                raise(ValueError('There should only be one entry per file index.'))
        
    def _build_files(self): 
        self.n_files = np.array([len(glob.glob(folder + '*.tim')) for folder in self.pth_data]).sum().astype(int)
        self.data = np.zeros((self.n_files, 2, self.n_samples), dtype=np.float32)
        self.file_index = np.zeros(self.n_files, dtype=int)
        
        n = 0 
        for folder_name in self.pth_data: 
            file_list = glob.glob(folder_name + '*.tim')
            
            for file_name in file_list: 
                self.file_index[n] = int(re.findall(r'\d+', file_name)[1])
                
                with open(file_name, 'rb') as f: 
                    data_sub = np.fromfile(f, dtype=np.float32)
                self.data[n, 0, :] = data_sub[np.arange(0, 2*self.n_samples, 2)]
                self.data[n, 1, :] = data_sub[np.arange(1, 2*self.n_samples, 2)]
                n += 1        
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        X, y = torch.from_numpy(self.data[index, :, :]), self.labels[index]
        return X, y 


class CSPLoader: 
    def __init__(self, pth_metadata:str, pth_data:list):
        self.dataset = CSPDataset(pth_metadata=pth_metadata, pth_data=pth_data)
    
    def split(self, split_ratio:float=0.8, num_workers:int=4, batch_size:int=128):
        train_size = int(split_ratio * self.dataset.n_files)
        test_size = self.dataset.n_files - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])
        
        dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        dataloader_valid = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)

        return dataloader_train, dataloader_valid
    