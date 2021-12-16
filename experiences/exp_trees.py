# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:28:59 2020
"""

import sys 
import os

sys.path.append(os.path.abspath("../IO"))
sys.path.append(os.path.abspath("../LDDMM"))
sys.path.append(os.path.abspath("../Optimization"))
sys.path.append(os.path.abspath("../tree_space"))

import torch
import time
import optimization
import numpy as np
import json

from IO_utils             import walklevel
from IO_parameters        import *
from import_export_vtk    import import_vtk,export_labeled_vtk,export_vector_field

from registration         import *
from keops_utils          import *

from new_tree_structures  import Points_threshold_from_singular, SingularToSegments

from tree_registration    import BHV_diffeo

from constants            import N_INTERP, THRSHLD_ORTHANT, THRSHLD_CONSTRUCTION, torchdtype, torchdeviceId

#To build the orthants

from VascularTree_class    import tree_from_vtk, print_recursive,recursive_tree_build_points, remove_labels_list, remove_leave_labels, get_recursive_max_length, VascularTree, plot_recursive, print_recursive
from BHV_class             import SubBHV
from main_geodesic         import InductiveMean, SturmMean
from visualization         import plot_tree_slide_depth
from phylogenetic_distance import check_same_topology

import numpy as np

############################## END OF THE IMPORTS #############################

# Cuda management
use_cuda,torchdeviceId,torchdtype,KeOpsdeviceId,KeOpsdtype,KernelMethod = TestCuda()

list_kernels = ['Sum4GaussKernel']

diffeo = True

from utils import walklevel


base = '../data/real_trees/'
sub  = '/all_branches/'


list_paths = []

for root, list_dirs, list_files in walklevel(base, 0):

    for i,f in enumerate(list_files):

        p = base+f # base+'Tree_{0}.vtk'.format(i) # 
        list_paths.append(p)


list_trees = []

for i,path_branch_real in enumerate(list_paths):
    
    #Load and resample a real case
    tree = tree_from_vtk(path_branch_real)
    tree.resample_tree(N_INTERP)
    list_trees.append(tree)


#Build all the possible orthants and the starting point for the optimization
#active_tree = -1 --> the sturm mean is used as initialization
orthants = SubBHV(list_trees, active_tree = -1)
n_topo = orthants.n_topo

print("Number of uniques topologies : ", orthants.n_topo)

singular_source_np, mask_topo_np = orthants.get_all_segpoints()

singular_source = torch.from_numpy(singular_source_np[:,:3]).detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
mask_topo       = torch.from_numpy(mask_topo_np).reshape(-1,n_topo).to(dtype=torchdtype, device=torchdeviceId)

print(mask_topo.sum(axis=0)) #should be 12

connections_source_np, mask_segments_np = orthants.compute_all_connections()

connections_source = torch.from_numpy(connections_source_np).reshape(-1,2).to(dtype=torch.long, device=torchdeviceId)
mask_segments      = torch.from_numpy(mask_segments_np).reshape(-1,n_topo).to(dtype=torchdtype, device=torchdeviceId)

print(mask_segments.sum(axis=0)) #should be 11

mask_topology_comparison = torch.zeros(connections_source.shape[0],mask_topo.shape[1]).to(dtype=torchdtype, device=torchdeviceId)

for i in range(n_topo):
    for s in range(connections_source.shape[0]):
        if not mask_segments[s,i] and not mask_topo[connections_source[s,1],i]:
            mask_topology_comparison[s,i] = 1
            
            
dictionnary_topology_comparison = {}


for i in range(n_topo):
    tmp = mask_segments.clone()
    test_diff = ((tmp-mask_segments[:,i].view(-1,1)).abs()).sum(dim=0)
    print(test_diff)
    l = []
    for j in range(n_topo):
        if test_diff[j] < mask_segments[:,0].shape[0]: 
            l.append(j)
    dictionnary_topology_comparison[i] = torch.tensor(l)
            
for k in dictionnary_topology_comparison.keys():
    print(k, " : ", dictionnary_topology_comparison[k])
            
            
#Setting the index of the current topology
ind_current_topo = 0
for i,t in enumerate(orthants.unique_topos):
    if check_same_topology(orthants.position, t):
        ind_current_topo = i
        break

print("Initial topology : ", ind_current_topo)

STS = SingularToSegments()
extremities_S, connections_S = STS.apply(singular_source, connections_source,  
                                         mask_topo, mask_segments, ind_current_topo)

print("Connections selected : ")
print(connections_S)
print("\n\n\n\n")
VS, CS = Points_threshold_from_singular(extremities_S, connections_S, n_interp=N_INTERP)

print("Source created")

path_config = './config_files/config_1.json'

with open(path_config,'r') as json_file:
    parameters = json.load(json_file)

gamma,factor,sigmaV,sigmaW,max_iter_steps,method,use_template = read_parameters(parameters)

print("Gamma : {0}, Factor : {1}, SigmaV : {2}, SigmaW : {3}, max_iter_steps : {4}, method : {5}".format(gamma,factor,sigmaV,sigmaW,max_iter_steps,method))

i = 11
          
num_target = (i)%len(list_paths)
path_branch_real = list_paths[num_target]

points, connections, labels = import_vtk(path_branch_real)
tree_target = tree_from_vtk(path_branch_real)
tree_target.resample_tree(N_INTERP)

#Setting the index of the current topology
ind_target_topo = -1
for i,t in enumerate(orthants.unique_topos):
    if check_same_topology(tree_target, t):
        ind_target_topo = i
        break

print("REAL TARGET TOPO : ", ind_target_topo)

VT_np, FT_np, labels_T_np = tree_target.Tree2Points()

print('¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤')
print("START \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

VT = torch.from_numpy(VT_np[:,:3]).to(dtype=torchdtype, device=torchdeviceId)

CT = torch.from_numpy(np.array(FT_np)).to(dtype=torch.long, device=torchdeviceId)
CS = CS.to(dtype=torch.long, device=torchdeviceId)

context = "registration_from_{1}_to_{2}/".format(THRSHLD_ORTHANT, ind_current_topo,ind_target_topo, THRSHLD_CONSTRUCTION) 

common_folder2save = './../results/real_trees/position_diffeo'+context

try_mkdir(common_folder2save)

common_folder2save2= common_folder2save + Params2FolderName(method,gamma,sigmaW,max_iter_steps,sigmaV)

#To save the results 
try_mkdir(common_folder2save2)

n_extremities = len(orthants.position.get_leaves_labels())+1 

ind_selected = BHV_diffeo(ind_current_topo, dictionnary_topology_comparison, 
                    mask_topology_comparison, mask_topo, 
                    mask_segments, singular_source, connections_source, 
                    n_extremities, VS, CS, VT, CT, 
                    common_folder2save2, parameters=parameters,
                    export=True) #set export to False if 

print("Registration done")

print("Resulting topology : ", ind_selected)
print("Ground truth topology : ", ind_target_topo)








