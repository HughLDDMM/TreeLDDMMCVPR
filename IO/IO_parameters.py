#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 18 09:49:05 2020
"""

import sys 
import os
sys.path.append(os.path.abspath("../IO"))
sys.path.append(os.path.abspath("../curves"))

import torch

from IO_utils                     import walklevel
from data_attachment              import CurvesDataloss,Compute_lengths
from keops_utils                  import *

import json

def try_mkdir(path):
    """
    return 0 if folder already exists, 1 if created.
    """
    r = 0

    try:
        os.mkdir(path)
        r = 1
    except OSError:
        pass

    return r

def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]
            
            

default_parameters = {  "default" : True,
                        "gamma" : 0.1,
                        "factor" : 1,
                        "sigmaV" : 100,
                        "sigmaW" : [100,25],
                        "max_iter_steps" : [100,500],
                        "method" : "ConstantNormalCycle",
                        "template" : False
                     }


def read_parameters(parameters = {}):
    """
    Read the parameters in a dictionary. If they are not provided, the default 
    parameters are taken.
    
    @param : parameters        : dictionary : supposed to contain : 
                                    - default : boolean: if True, set the paramters to default. 
                                    - method  : string : data attachment method ("ConstantNormalCycle", "LinearNormalCycle", 
                                                                                "CombineNormalCycle", "PartialVarifold",
                                                                                "Varifold")
                                    - factor  : float  : the factor scale to homogenize the losses if the scale changes. 
                                    - sigmaV  : float  : scale of the diffeomorphism.
                                    - sigmaW  : list of float : the different scales of the data attachment term. 
                                    - max_iter_steps : list of integers : must be same size as sigmaW, the number of iteration 
                                                                          per data attachment scale. 
                                    - template : boolean : if we want to build a template from the registration to all the targets. 
                                    
    """

    if("gamma" in parameters.keys()):
        gamma = parameters["gamma"]
    else:
        gamma = torch.tensor([default_parameters["gamma"]], dtype=torchdtype, device=torchdeviceId)
    if("factor" in parameters.keys()):
        factor = parameters["factor"]
    else:
        factor = default_parameters["factor"]
    if("sigmaV" in parameters.keys()):
        scaleV = factor*parameters["sigmaV"]
    else:
        scaleV = factor*default_parameters["sigmaV"]
    sigmaV = torch.tensor([scaleV], dtype=torchdtype, device=torchdeviceId)
    
    if("sigmaW" in parameters.keys()):
        sigmaW = [factor*torch.tensor([sigW], dtype=torchdtype, device=torchdeviceId) for sigW in parameters["sigmaW"]]
    else:
        sigmaW = [factor*torch.tensor([sigW], dtype=torchdtype, device=torchdeviceId) for sigW in default_parameters["sigmaW"]]
    
    if("max_iter_steps" in parameters.keys()):
        max_iter_steps = parameters["max_iter_steps"]
    else:
        max_iter_steps = default_parameters["max_iter_steps"]
    if(len(max_iter_steps)!=len(sigmaW)):
        max_iter_steps = [50 for s in sigmaW]
        
    if("method" in parameters.keys()):
        method = parameters["method"]
    else:
        method = default_parameters["method"]
    
    if("template" in parameters.keys()):
        use_template=parameters["template"]
    else:
        use_template=default_parameters["template"]

    return gamma,factor,sigmaV,sigmaW,max_iter_steps,method,use_template


def Params2FolderName(method,gamma,sigmaW,max_iter_steps,sigmaV):
    """
    Converts parameters into folder name, to standardize through scripts
    """

    name = '{0}_SigmaV_{4}_gamma_{1}_W{2}_{3}it/'.format(method,gamma,[int(w) for w in sigmaW],int(max_iter_steps[0]),int(sigmaV))    

    return name


def FolderName2Params(foldername):
    """

    Converts foldername to parameters that were used to generate it, to standardize through scripts
    """  

    method, _Sv, sigmaV, _gma, gamma, to_split_att, to_split_it = foldername.split('_')

    sigmaW         = to_split_att[1:]
    max_iter_steps = to_split_it[:-3]

    print( method,gamma,sigmaW,max_iter_steps,sigmaV )

    resW  = [float(s) for s in sigmaW.strip('][').split(', ')]
    sigmaW = [torch.tensor([sigW], dtype=torchdtype, device=torchdeviceId) for sigW in resW]

    max_iter_steps = int(max_iter_steps)
    sigmaV = torch.tensor([float(sigmaV)], dtype=torchdtype, device=torchdeviceId)    

    return method, int(gamma), sigmaW, max_iter_steps, sigmaV


def save_parameters(gamma, factor, sigmaV, sigmaW, max_iter_steps, method, use_template, filename, path2save, Kv = 'Default', comments = ''):
    """

    """
    dic = {}

    if Kv == 'Default' :
        print("Using default Kv as parameter : GaussKernel")
        Kv = 'GaussKernel'

    dic["gamma"]          = gamma
    dic["factor"]         = factor
    dic["sigmaV"]         = int(sigmaV)
    dic["sigmaW"]         = [int(w) for w in sigmaW]
    dic["max_iter_steps"] = max_iter_steps
    dic["method"]         = method
    dic["template"]       = use_template
    dic["Kv"]             = Kv
    dic["comments"]       = comments

    with open(path2save+'/'+filename+'.json', 'w') as f:
        json.dump(dic, f, indent = 4)
    f.close()
    

    return 0

