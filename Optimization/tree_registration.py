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

import new_tree_optimization
import optimization

from IO_parameters                import read_parameters, try_mkdir
from data_attachment              import CurvesDataloss
from registration                 import Rigidloss, LDDMMPositionLoss, LDDMMloss, Flow, Shooting
from linear_functions             import RawRegistration, RigidDef
from keops_utils                  import Sum4GaussKernel, GaussKernel
from import_export_vtk            import export_labeled_vtk

from new_tree_structures      import Points_threshold_from_singular, SingularToSegments, Control_from_singular
from constants                import *







def register_from_bary(template, template_connections,
                        target, target_connections,
                        folder2save,
                        parameters={"default" : True},
                        list_kernels = ['Sum4GaussKernel', 'GaussKernel']):
    """
    Perform the registration of a source onto a target with an initialization as barycenter registration.
    
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

    gamma,factor,sigmaV,sigmaW,max_iter_steps,method,use_template = read_parameters(parameters)
    
    print("PARAMETERS : ")
    print("Gamma : {0}, Factor : {1}, SigmaV : {2}, SigmaW : {3}, max_iter_steps : {4}, method : {5}".format(gamma,factor,sigmaV,sigmaW,max_iter_steps,method))
    print()
    resum_optim = {}

    #To save the results
    try_mkdir(folder2save)    

    decalage     =  RawRegistration(template,target, use_torch=True) # register_ICP(template, target)   
    matrix_rigid =  np.eye(4) 

    #Stores the translation in the rigid matrix transformation for homogeneous output.
    matrix_rigid[0,3] += decalage[0]
    matrix_rigid[1,3] += decalage[1]
    matrix_rigid[2,3] += decalage[2]

    #vertices
    template = template.clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
    target = target.detach().to(dtype=torchdtype, device=torchdeviceId)
    #faces
    template_connections = template_connections.detach().to(dtype=torch.long, device=torchdeviceId)
    target_connections = target_connections.detach().to(dtype=torch.long, device=torchdeviceId)
    
    #The kernel deformation definition
    Kv = Sum4GaussKernel(sigma=sigmaV)
    folder2save_sub = folder2save
    folder_resume_results = folder2save_sub+'dict_resume_opt/'

    try_mkdir(folder_resume_results)

    p0 = torch.zeros(template.shape, dtype=torchdtype, device=torchdeviceId, requires_grad=True)

    #Save the rigid deformation and deformed template
    np.save(folder2save_sub +'/rigid_def.npy', template.detach().cpu().numpy())
    np.save(folder2save_sub+'/rigid2apply.npy',matrix_rigid) 
    
    for j,sigW in enumerate(sigmaW):
        print("The diffeomorphic part")
        print("Scale : ", sigW)
        tensor_scale = sigW

        ######### Loss to update in both case
        dataloss_att = CurvesDataloss(method,template_connections, target, target_connections, tensor_scale, VS = template).data_attachment()      
            

        ######### The diffeo part
        loss = LDDMMloss(Kv,dataloss_att,sigmaV, gamma=gamma)

        dict_opt_name = 'scale_'+str(int(tensor_scale))+'_sum4kernels_'+str(int(sigmaV.detach().cpu().numpy()))
        p0,nit,total_time = optimization.opt(loss, p0, template, max_iter_steps[j], folder_resume_results, dict_opt_name)
        p_i,deformed_i = Shooting(p0, template, Kv)

        resum_optim[str(sigW)]="Diffeo Iter : "+str(nit)+", total time : "+str(total_time)

        if(tensor_scale>=1):
            filesavename = 'Scale_'+str(int(tensor_scale))
        else:
            filesavename = 'Scale_'+str(int(10*tensor_scale))+'div10_'

        qnp_i = deformed_i.detach().cpu().numpy()
        
        print("Saving in : ", folder2save_sub)
        template_labels = np.ones(qnp_i.shape[0])
        export_labeled_vtk(qnp_i,template_connections,template_labels,filesavename,folder2save_sub)

        p0_np = p0.detach().cpu().numpy()

        np.save(folder2save_sub +'/template_' + filesavename +'.npy', template.detach().cpu().numpy())
        np.save(folder2save_sub+'/Momenta_'+filesavename+'.npy',p0_np)
        np.save(folder2save_sub+'/'+filesavename+'.npy',qnp_i)

        #Save the optimization informations
        f = open(folder2save_sub+'/norot_resume_optimization.txt',"w")
        for key,value in resum_optim.items():    
            f.write(key)
            f.write(" : ")
            f.write(value)
            f.write('\n')
        f.close()

        np.save(folder2save_sub+'/target.npy',target.detach().cpu().numpy())
        np.save(folder2save_sub+'/momenta2apply.npy',p0_np) 
        np.save(folder2save_sub+'/template2apply.npy',template.detach().cpu().numpy()) 

    return 0






def BHV_diffeo( ind_current_topo, dictionnary_topology_comparison, 
                mask_topo_comparison, mask_topo, mask_segments, 
                singular_template, singular_connections, n_leaves,
                template, template_connections,
                target, target_connections,
                folder2save,
                parameters={"default" : True}, n_interp = N_INTERP, export = False, target_curves=None):
    """
    Perform the registration of a source onto a target with an initialization as barycenter registration.
    
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

    gamma,factor,sigmaV,sigmaW,max_iter_steps,method,use_template = read_parameters(parameters)
    
    print("PARAMETERS : ")
    print("Gamma : {0}, Factor : {1}, SigmaV : {2}, SigmaW : {3}, max_iter_steps : {4}, method : {5}".format(gamma,factor,sigmaV,sigmaW,max_iter_steps,method))
    print()
    resum_optim = {}

    #method = 'Varifold'

    #To save the results
    try_mkdir(folder2save)    

    folder2save_sub = folder2save 
    folder_resume_results = folder2save_sub+'dict_resume_opt/'

    try_mkdir(folder_resume_results)

    target_deformed = target
    target_np = target.detach().cpu().numpy()
    
    np.save(folder2save_sub+'/target.npy',target_np)
    target_labels = 100+np.ones(target.shape[0])
    export_labeled_vtk(target_np,target_connections,target_labels,"target_to_see",folder2save_sub)

    Kv = Sum4GaussKernel(sigma=sigmaV)

    n_branches = 2*n_leaves-3 #because we deal with binary trees fully resolved
    n_ctrl_pts = (n_interp-2)*n_branches #- 2*n_leaves-2
    
    p0 = torch.zeros(( n_ctrl_pts, template.shape[1]), dtype=torchdtype, device=torchdeviceId, requires_grad=True)
     
    tmp        = SingularToSegments()
    ToSegments = tmp.apply                                          
    extremities, connections = ToSegments(singular_template, singular_connections, 
                                          mask_topo, mask_segments, ind_current_topo)
    
    decalage = RawRegistration(extremities,target, use_torch=True)  
    singular_template.data += decalage
    points, connections_points = Points_threshold_from_singular(extremities, connections, n_interp=n_interp)

    quat0 = torch.tensor([1,0,0,0,0,0,0], dtype=torchdtype, device=torchdeviceId, requires_grad=True)
    deformed_rigid = template.clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)

    ######## THE RIGID PART USING QUATERNIONS #######
    sigW = sigmaW[0] #for i,sigW in enumerate(sigmaW):

    tensor_scale = torch.tensor([40]).to(dtype=torchdtype, device=torchdeviceId)

    ######### Loss to update in both case
    dataloss_att = CurvesDataloss("Varifold", connections_points, target, target_connections, tensor_scale).data_attachment() 

    loss = Rigidloss(dataloss_att, rotate=False)
    quat0,nit,total_time = optimization.rigid_opt(loss, quat0, points.clone().detach().to(dtype=torchdtype, device=torchdeviceId), max_iter_steps[0])
    
    resum_optim[str(sigW)]="Rigid Iter : "+str(nit)+", total time : "+str(total_time)
    
    deformed_rigid = RigidDef(singular_template, quat0, rotate=False)

    singular_template = deformed_rigid.clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
         
        
    extremities, connections = ToSegments(singular_template, singular_connections, 
                                          mask_topo, mask_segments, ind_current_topo)
    
    points2, connections_points2 = Points_threshold_from_singular(extremities, connections, n_interp=n_interp)
    
    print("DIFFERENCE : ", (points2 - RigidDef(points, quat0, rotate=False)).norm())
    
    for j,sigW in enumerate(sigmaW):
        print('\n \n \n \n \n \n')
        print("The diffeomorphic part")
        print("Scale : ", sigW)
        tensor_scale = sigW
        #template = torch.clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)

        folder_resume_results = folder2save_sub+'dict_resume_opt'+str(int(sigW.data))+'/'
        
        #p0 = torch.zeros(( n_ctrl_pts, template.shape[1]), dtype=torchdtype, device=torchdeviceId, requires_grad=True)

        ######### Loss to update in both case
        if method == "OT":
            dataloss_att = CurvesDataloss(method,template_connections, target_deformed, target_connections, tensor_scale, target_curves_map = target_curves).data_attachment()
        else:
            dataloss_att = CurvesDataloss(method,template_connections, target_deformed, target_connections, tensor_scale).data_attachment()

        ######### The diffeo part
        loss = LDDMMPositionLoss(Kv,dataloss_att,sigmaV, gamma=gamma)

        dict_opt_name = 'scale_'+str(int(tensor_scale))+'_sum4kernels_'+str(int(sigmaV.detach().cpu().numpy()))
                                                                                   
        p0, singular_template, ind_current_topo, nit, total_time = new_tree_optimization.boundary_opt(loss, p0, 
                                                                                            ind_current_topo, dictionnary_topology_comparison, 
                                                                                            mask_topo_comparison, mask_topo, mask_segments, 
                                                                                            singular_template, singular_connections, sigW, Kv, n_leaves, n_interp,
                                                                                            max_iter_steps[j], folder_resume_results, dict_opt_name, export = export)



        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        extremities, connections = ToSegments(singular_template, singular_connections, 
                                             mask_topo, mask_segments, ind_current_topo)
        
        points, connections_points = Points_threshold_from_singular(extremities, connections, n_interp=n_interp)

        q0 = Control_from_singular(extremities, connections, n_leaves, n_interp=n_interp)
        
        print(q0.shape, p0.shape, points.shape)
        
        deformed_i, p_i, q_i = Flow(points, p0, q0, Kv)

        deformed_np = deformed_i.detach().cpu().numpy()
        connections_np = connections_points.detach().cpu().numpy()
                
        print("Saving in : ", folder2save_sub)
        
        if(tensor_scale>=1):
            filesavename = 'Scale_'+str(int(tensor_scale))+'_topo_'+str(ind_current_topo)
        else:
            filesavename = 'Scale_'+str(int(10*tensor_scale))+'div10_topo_'+str(ind_current_topo)
        
        template_labels = np.ones(points.shape[0])
        tree_labels     = np.ones(deformed_np.shape[0])
        export_labeled_vtk(points.detach().cpu().numpy(),connections_np,template_labels,"template_"+filesavename,folder2save_sub)
        export_labeled_vtk(deformed_np,connections_np,tree_labels,filesavename,folder2save_sub)

        np.save(folder2save_sub +'/template_' + filesavename +'.npy', points.detach().cpu().numpy())
        np.save(folder2save_sub +'/control_points_' + filesavename +'.npy', q0.detach().cpu().numpy())

    #Save the optimization informations
    f = open(folder2save_sub+'/coordinates_resume_optimization.txt',"w")
    for key,value in resum_optim.items():    
        f.write(key)
        f.write(" : ")
        f.write(value)
        f.write('\n')
    f.close()

    target_np = target_deformed.detach().cpu().numpy()
    
    np.save(folder2save_sub+'/target.npy',target_np)
    np.save(folder2save_sub+'/template2apply.npy',template.detach().cpu().numpy()) 

    target_labels = 100+np.ones(target.shape[0])
    export_labeled_vtk(target_np,target_connections,target_labels,"target_to_see",folder2save_sub)

    template_labels = np.ones(template.shape[0])
    export_labeled_vtk(template.detach().cpu().numpy(),template_connections.detach().cpu().numpy(),template_labels,"template_to_see",folder2save_sub)

    return ind_current_topo
