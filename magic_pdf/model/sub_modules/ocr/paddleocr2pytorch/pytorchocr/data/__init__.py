from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import numpy as np
# import paddle
import signal
import random

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))


import copy
# from paddle.io import Dataset, DataLoader, BatchSampler, DistributedBatchSampler
# import paddle.distributed as dist

from pytorchocr.data.imaug import transform, create_operators
# from pytorchocr.data.simple_dataset import SimpleDataSet
# from pytorchocr.data.lmdb_dataset import LMDBDateSet

