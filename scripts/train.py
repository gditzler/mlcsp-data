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

from lightning.pytorch.loggers import WandbLogger
from csp.base import CSPModel
from csp.data import CSPLoader

opts = {
    'lr': 1e-3,
    'optimizer': 'adam', 
    'split': 0.8, 
    'num_workers': 4,
    'batch_size': 256,
    'path_data': [
        'data/PM_One_Batch_1/', 
        'data/PM_One_Batch_2/', 
        'data/PM_One_Batch_3/',
        'data/PM_One_Batch_4/',  
    ],
    'path_metadata': 'data/PM_single_truth_10000.csv',
}


if __name__ == '__main__': 
    loader = CSPLoader(pth_data=opts['path_data'], pth_metadata=opts['path_metadata'])
    dataloader_train, dataloader_valid = loader.split(
        split_ratio=opts['split'], 
        num_workers=opts['num_workers'], 
        batch_size=opts['batch_size']
    )
    # logger = WandbLogger(project="MLCSP", log_model=True)
    # logger.log_hyperparams(opts)
    