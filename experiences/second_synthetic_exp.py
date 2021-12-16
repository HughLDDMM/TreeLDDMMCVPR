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

from tree_registration    import BHV_diffeo

from constants import N_INTERP, THRSHLD_ORTHANT, THRSHLD_CONSTRUCTION, torchdtype, torchdeviceId

from import_export_vtk            import import_vtk



path_dic = "./config_files/config_synthetics.json" 

common_folder2save = '../results_paper/second_synthetic_example/'
try_mkdir(common_folder2save)


with open(path_dic,'r') as json_file:
    parameters = json.load(json_file)

gamma,factor,sigmaV,sigmaW,max_iter_steps,method,use_template = read_parameters(parameters)

common_folder2save2= common_folder2save + Params2FolderName(method,gamma,sigmaW,max_iter_steps,sigmaV)

#To save the results 
try_mkdir(common_folder2save2)

n_topo = 3

singular_source = torch.tensor([[0, 0, 0],
                                [0, -4, 0],
                                [-4, -7, 0],
                                [-2, -7, 0],
                                [2, -7, 0],
                                [4, -7, 0],
                                [0, -4, 0],
                                [0, -4, 0],
                                [0, -4, 0],
                                [0, -4, 0],
                              ]).reshape(-1,3).detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)


connections_source = torch.tensor([[0,1],
                                   [1,6],
                                   [6,7],
                                   [7,2],
                                   [7,3],
                                   [6,4],
                                   
                                   [1,2],
                                   
                                   [1,7],
                                   [1,8],
                                   [8,4],
                                   [8,5],
                                   [1,5], 
                                   [1,9],
                                   [9,3],
                                   [9,8]
                                  ]).reshape(-1,2).to(dtype=torch.long, device=torchdeviceId)
   
#
#


  
singular_target = torch.tensor([[0, 0, 0],
                                [0, -4, 0],
                                [-2, -6, 0],
                                [-5, -8, 0],
                                [-7, -10, 0],
                                [-2, -10, 0],
                                [0, -8, 0],
                                [2, -6, 0]
                              ]).reshape(-1,3).detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)

#last point working case : [6,-9,0]
                      
connections_target = torch.tensor([[0,1],
                                   [1,2],
                                   [2,3],
                                   [3,4],
                                   [3,5],
                                   [2,6],
                                   [1,7]
                                  ]).reshape(-1,2).to(dtype=torch.long, device=torchdeviceId)


mask_topo = torch.ones(singular_source.shape[0],3).to(dtype=torchdtype, device=torchdeviceId)
mask_topo[6,1]=0
mask_topo[6,2]=0
mask_topo[7,2]=0
mask_topo[8,0]=0
mask_topo[9,0]=0
mask_topo[9,1]=0

mask_segments = torch.ones(connections_source.shape[0],mask_topo.shape[1]).to(dtype=torchdtype, device=torchdeviceId)



mask_segments[6,0] = 0
mask_segments[7,0] = 0
mask_segments[8,0] = 0
mask_segments[9,0] = 0
mask_segments[10,0] = 0
mask_segments[12,0] = 0
mask_segments[13,0] = 0
mask_segments[14,0] = 0
  

mask_segments[1,1] = 0
mask_segments[2,1] = 0
mask_segments[5,1] = 0
mask_segments[6,1] = 0
mask_segments[11,1] = 0
mask_segments[12,1] = 0
mask_segments[13,1] = 0
mask_segments[14,1] = 0


mask_segments[1,2] = 0
mask_segments[2,2] = 0
mask_segments[3,2] = 0
mask_segments[4,2] = 0
mask_segments[5,2] = 0
mask_segments[7,2] = 0
mask_segments[8,2] = 0
mask_segments[11,2] = 0


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
extremities_S, connections_S = STS.apply(singular_source, connections_source, mask_topo, mask_segments, ind_current_topo)

print("EXTREMITIES : ", extremities_S, " \n Connections : ", connections_S)

VS, CS = Points_threshold_from_singular(extremities_S, connections_S, n_interp=N_INTERP)


VT, CT, target_labels = import_vtk('../data/synthetic_target/target_to_see_more_difficult.vtk')
#VT, CT = Points_from_singular(singular_target, connections_target, n_interp=N_INTERP)

print('¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤')
print("START \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")


VT= torch.from_numpy(VT).to(dtype=torchdtype, device=torchdeviceId)
CT= torch.from_numpy(CT).to(dtype=torch.long, device=torchdeviceId)

CS = CS.to(dtype=torch.long, device=torchdeviceId)

folder2save = common_folder2save2
try_mkdir(folder2save)

export = True

n_leaves = 5

BHV_diffeo(ind_current_topo, dictionnary_topology_comparison, 
           mask_topology_comparison, mask_topo, mask_segments, 
           singular_source, connections_source, n_leaves,
           VS, CS, VT, CT, 
            folder2save, parameters=parameters, export = True)
                     
print("Registration done")






