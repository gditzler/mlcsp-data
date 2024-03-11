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

import torch 
import torch.nn as nn 

class BluePengiun(nn.Module): 
    def __init__(self, num_classes:int=8):
        super(BluePengiun, self).__init__()
        self.num_classes = num_classes
        
        def conv_block(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0)
            )
         
        self.fe = nn.Sequential(
            conv_block(2, 32, 16, 1, 0), 
            conv_block(32, 64, 32, 1, 0),
            conv_block(64, 96, 64, 1, 0), 
        )
        self.relu = nn.ReLU()

        self.fc1 = nn.LazyLinear(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, self.num_classes)
    
    def forward(self, x):
        
        x = self.fe(x)
        
        # Global Average Pooling (GAP)
        x = torch.mean(x, dim=2)  # Calculate the mean along the spatial dimension (width)
        # x = x.view(x.size(0), -1, x.size(1))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x) 
        return x