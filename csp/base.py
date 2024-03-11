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
from torch import optim
import torch.nn as nn
import torchmetrics

import lightning as L 


class CSPModel(L.LightningModule): 
    def __init__(self, model, opts:dict=None):
        super().__init__() 
        self.model = model
        
        if opts is None: 
            opts = {
                'optimizer': 'adam', 
                'lr': 1e-3, 
                'num_classes': 9, 
            }
        self.opts = opts
        self.acc1 = torchmetrics.Accuracy(task='multiclass', num_classes=opts['num_classes'], top_k=1)
        self.acc2 = torchmetrics.Accuracy(task='multiclass', num_classes=opts['num_classes'], top_k=2)
        # self.auc = torchmetrics.AUROC(num_classes=num_classes, task='multiclass')
        
        
    def forward(self, x): 
        return self.model(x) 
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        phat = self(x)
        loss = nn.functional.cross_entropy(phat, y.type(torch.long))
        self.log('train_loss', loss)
        self.log('train_acc1', self.acc1(phat, y.type(torch.long)))
        self.log('train_acc2', self.acc2(phat, y.type(torch.long)))
        # self.log('train_auc', self.auc(phat, y))
        return loss
    
    def validation_step(self, batch, batch_index):
        x, y = batch
        phat = self(x)
        loss = nn.functional.cross_entropy(phat, y.type(torch.long))
        self.log('valid_loss', loss)
        self.log('valid_acc1', self.acc1(phat, y.type(torch.long)))
        self.log('valid_acc2', self.acc2(phat, y.type(torch.long)))
        # self.log('valid_auc', self.auc(phat, y))
        return loss
    
    def test_step(self, batch, batch_index):
        x, y = batch
        phat = self(x)
        loss = nn.functional.cross_entropy(phat, y.type(torch.long))
        self.log('test_loss', loss)
        self.log('test_acc1', self.acc1(phat, y.type(torch.long)))
        self.log('test_acc2', self.acc2(phat, y.type(torch.long)))
        # self.log('test_auc', self.auc(phat, y))
        return loss

    def configure_optimizers(self):
        if self.opts['optimizer'] == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.opts['lr'])
        elif self.opts['optimizer'] == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.opts['lr'])
        elif self.opts['optimizer'] == 'nadam':
            optimizer = optim.Nadam(self.parameters(), lr=self.opts['lr'])
        else:
            raise ValueError(f'Optimizer not supported: {self.opts["optimizer"]}')
        return optimizer
 


