# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:28:59 2020
"""

import sys 
import os
sys.path.append(os.path.abspath("../IO"))
sys.path.append(os.path.abspath("../LDDMM"))
sys.path.append(os.path.abspath("../Optimization"))

import torch
import numpy as np
import json

from IO_parameters                import read_parameters, try_mkdir, Params2FolderName

from new_tree_structures      import Points_threshold_from_singular, SingularToSegments

from tree_registration    import register_from_bary, BHV_diffeo

from constants import N_INTERP, torchdtype, torchdeviceId


path_dic = "./config_files/config_synthetics.json" 

base = '../results_paper/first_synthetic_example/'

try_mkdir(base)

common_folder2save = base + '/LDDMM_TreeSpace/' 

try_mkdir(common_folder2save)

with open(path_dic,'r') as json_file:
    parameters = json.load(json_file)

gamma,factor,sigmaV,sigmaW,max_iter_steps,method,use_template = read_parameters(parameters)

common_folder2save2= common_folder2save + Params2FolderName(method,gamma,sigmaW,max_iter_steps,sigmaV)

#To save the results 
try_mkdir(common_folder2save2)

n_topo = 2

singular_source = torch.tensor([[0, 0, 0],
                                [0, -4, 0],
                                [0, -4, 0],
                                [-5, -8, 0],
                                [0, -8, 0],
                                [2, -6, 0],
                                [6, -9, 0]
                              ]).reshape(-1,3).detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)


connections_source = torch.tensor([[0,1],
                                   [1,2],
                                   [2,3],
                                   [2,4],
                                   [1,5],
                                   [5,4],
                                   [5,6],
                                   [1,3],
                                   [1,6]
                                  ]).reshape(-1,2).to(dtype=torch.long, device=torchdeviceId)

  
  
singular_target = torch.tensor([[0, 0, 0],
                                [0, -4, 0],
                                [-2, -6, 0],
                                [-5, -8, 0],
                                [-0, -8, 0],
                                [5, -8, 0]
                              ]).reshape(-1,3).detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)

#last point working case : [6,-9,0]
                      
connections_target = torch.tensor([[0,1],
                                   [1,2],
                                   [2,3],
                                   [2,4],
                                   [1,5]
                                  ]).reshape(-1,2).to(dtype=torch.long, device=torchdeviceId)


mask_topo = torch.ones(singular_source.shape[0],2).to(dtype=torchdtype, device=torchdeviceId)
mask_topo[5,0]=0
mask_topo[2,1]=0


mask_segments = torch.ones(connections_source.shape[0],mask_topo.shape[1]).to(dtype=torchdtype, device=torchdeviceId)
mask_segments[4,0] = 0
mask_segments[5,0] = 0
mask_segments[6,0] = 0
mask_segments[7,0] = 0


mask_segments[1,1] = 0
mask_segments[2,1] = 0
mask_segments[3,1] = 0
mask_segments[-1,1] = 0


mask_topology_comparison = torch.zeros(connections_source.shape[0],mask_topo.shape[1]).to(dtype=torchdtype, device=torchdeviceId)

for i in range(mask_topo.shape[1]):
    for s in range(connections_source.shape[0]):
        if not mask_segments[s,i] and not mask_topo[connections_source[s,1],i]:
            mask_topology_comparison[s,i] = 1
            



dictionnary_topology_comparison = {}

print(mask_segments[:,0].shape)

for i in range(n_topo):
    tmp = mask_segments.clone()
    test_diff = ((tmp-mask_segments[:,i].view(-1,1)).abs()).sum(dim=0)
    print(test_diff)
    l = []
    for j in range(n_topo):
        if test_diff[j] < mask_segments[:,0].shape[0]: # 16: # 
            l.append(j)
    dictionnary_topology_comparison[i] = torch.tensor(l)
            
for k in dictionnary_topology_comparison.keys():
    print(k, " : ", dictionnary_topology_comparison[k])


            
ind_current_topo = 1
STS = SingularToSegments()
extremities_S, connections_S = STS.apply(singular_source, connections_source,  mask_topo, mask_segments, ind_current_topo)
VS, CS = Points_threshold_from_singular(extremities_S, connections_S, n_interp=N_INTERP)    
VT, CT = Points_threshold_from_singular(singular_target, connections_target, n_interp=N_INTERP)

print('¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤')
print("START \n\n\n\n\n\n\n\n\n\n\n")

CS = CS.to(dtype=torch.long, device=torchdeviceId)

folder2save = common_folder2save2
try_mkdir(folder2save)

n_leaves = 4

BHV_diffeo(ind_current_topo, dictionnary_topology_comparison, 
            mask_topology_comparison, mask_topo, mask_segments, 
            singular_source, connections_source, n_leaves,
            VS, CS, VT, CT, 
            folder2save, parameters=parameters, export = True)

print('Classic LDDMM Registration')
print("START \n\n\n\n\n\n\n\n\n\n\n")

config_varif = '/home/pantonsanti/CIFRE/VascularTreeLabeling/experiences/config_files/classique/config_0.json'

common_folder2save = base + '/Classic_LDDMM/'

try_mkdir(common_folder2save)

with open(config_varif,'r') as json_file:
    parameters = json.load(json_file)

gamma,factor,sigmaV,sigmaW,max_iter_steps,method,use_template = read_parameters(parameters)

folder2save= common_folder2save + Params2FolderName(method,gamma,sigmaW,max_iter_steps,sigmaV)

parameters["method"] = "Varifold"

#To save the results 
try_mkdir(folder2save)

register_from_bary(VS, CS,
                        VT, CT,
                        folder2save,
                        parameters=parameters)

print("Registration done")






