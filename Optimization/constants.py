# -*- coding: utf-8 -*-
"""
Created on Wed Sep 8 13:49:04 2021
"""

import sys 
import os

sys.path.append(os.path.abspath("../LDDMM"))

from keops_utils     import TestCuda

use_cuda,torchdeviceId,torchdtype,KeOpsdeviceId,KeOpsdtype,KernelMethod = TestCuda()

THRSHLD_ORTHANT = 40
THRSHLD_CONSTRUCTION = 0
THRSHLD_LENGTH  = 1e-1
N_INTERP        = 50

