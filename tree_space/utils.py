# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 14:12:12 2021
"""

import math
from scipy.ndimage import zoom

import os

def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]
            

def BranchPath(Path2BranchesRep,current_branch):
    
    branch_path = Path2BranchesRep+'/branch'
    digits = int(math.log10(current_branch))+1 #number of digits, for the branch path

    for ind_zero in range(4-digits):
        branch_path+='0'    

    branch_path+=str(current_branch)+'.txt'
    
    return branch_path


def resample_curve(curve, n_points, method = 'fourier'):
    """
    Resample a curve using zoom method. 
    """
    curve[curve[:,:]==0]=1e-10
    
    rescurve = zoom(curve, (n_points/curve.shape[0],1))

    rescurve[0,:]=curve[0,:]
    rescurve[-1,:]=curve[-1,:]
    
    return rescurve
