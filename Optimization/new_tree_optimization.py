######################### perform optimization ##############################

import numpy as np
import time
import pickle
import sys 
import os

import torch
from torch import autograd
from scipy.optimize import minimize

sys.path.append(os.path.abspath("../IO"))
sys.path.append(os.path.abspath("../LDDMM"))
from import_export_vtk import export_labeled_vtk, export_points

from new_tree_structures import SingularToSegments, Points_threshold_from_singular, FindParent, SelectAvailableTopo
from new_tree_structures import GoToBoundary, GoToBoundaryWithMomenta, Control_from_singular, RearangeMomenta

from keops_utils     import TestCuda
from registration import Flow


from constants import N_INTERP, THRSHLD_ORTHANT

params_opt=dict({"lr" : 1,"maxcor" : 10, "gtol" : 1e-9, "tol" : 1e-1, "use_scipy" : False, "method" : 'SLSQP'})
use_cuda,torchdeviceId,torchdtype,KeOpsdeviceId,KeOpsdtype,KernelMethod = TestCuda()


def Regularization(singular_points, singular_connections, 
                   mask_topo, mask_segments, ind_current_topo):
    """
    To prevent from moving far from the star tree (not used in CVPR 2022)
    """
    activated_segments = mask_segments[:,ind_current_topo]
    
    R = 0
    
    for i, con in enumerate(singular_connections):
            
        start = singular_points[con[0],:]
        end   = singular_points[con[1],:]
                
        if activated_segments[i] and (mask_topo[con[1],:]==0).any():
            
            R += (end-start).norm(p=2).square()
    
    return R.sqrt()


def Verification_LDDMM(loss, momenta, ind_current_topo, mask_topo, mask_segments, 
                      singular_points, singular_connections, list_test, list_available, 
                      n_leaves, n_interp = N_INTERP, epsilon = 1e-8):
    """
    
    """
    
    list_test = []
    
    ToSegments2 = SingularToSegments().apply
    
    tmp = singular_points.clone() 
    
    p0 = momenta.clone()

    extremities, connections   = ToSegments2(tmp, singular_connections, 
                                           mask_topo, mask_segments, ind_current_topo, 
                                           list_test,  list_available)
    points, connections_points = Points_threshold_from_singular(extremities, connections, n_interp=n_interp)
    
    q0                         = Control_from_singular(extremities, connections, n_leaves, n_interp=n_interp)

    L1 = loss(points, p0, q0, connections_points)
    
    new_points = tmp + epsilon * (singular_points.grad)
    new_momenta = p0 + 0 * (momenta.grad)
    
    extremities2, connections2   = ToSegments2(new_points, singular_connections, 
                                           mask_topo, mask_segments, ind_current_topo, 
                                           list_test,  list_available)
    points2, connections_points2 = Points_threshold_from_singular(extremities2, connections2, n_interp=n_interp)
    q02                          = Control_from_singular(extremities2, connections2, n_leaves, n_interp=n_interp)
     
    L2 = loss(points2, new_momenta, q02, connections_points2)
    
    res = (L2-L1)/epsilon
    
    print("RES : ", res)
    
    n = singular_points.grad.norm(p=2)**2
    
    print("VERIFICATION TEST POINTS: ", (res - n)/n)
    
    new_points = tmp + 0 * (singular_points.grad)
    new_momenta = p0 + epsilon * (momenta.grad)
    
    extremities2, connections2   = ToSegments2(new_points, singular_connections, 
                                           mask_topo, mask_segments, ind_current_topo, 
                                           list_test,  list_available)
    points2, connections_points2 = Points_threshold_from_singular(extremities2, connections2, n_interp=n_interp)
    q02                          = Control_from_singular(extremities2, connections2, n_leaves, n_interp=n_interp)
     
    L3 = loss(points2, new_momenta, q02, connections_points2)
    
    res2 = (L3-L1)/epsilon
    m = momenta.grad.norm(p=2)**2
    print("VERIFICATION TEST MOMENTA: ", (res2 - m)/m)
    
    return res, res2





def boundary_opt(loss,  p0, ind_current_topo, dictionnary_topology_comparison,
                         mask_topo_comparison, mask_topo, mask_segments, 
                         singular_points, singular_connections, sigmaW, Kv, n_leaves,
                         n_interp = N_INTERP, maxiter = 100, folder2save = '',savename = '', export = False):
    """
    Optimization function calling either scipy or torch method. 
    singular_points is the variable to optimize, the singular points (extremities, bifurcations)
    of a template tree composed of segments, but with possibly contracted branches.
    """
    
    lr                            = params_opt["lr"] 
    gtol                          = params_opt["gtol"]
    tol                           = params_opt["tol"]
    max_eval                      = 10

    boundary_opt.list_test        = [ind_current_topo]
    boundary_opt.list_available   = [ind_current_topo]
    boundary_opt.last_topo_proj   = []
    
    boundary_opt.allow_projection = True
    boundary_opt.went2boundary    = False

    boundary_opt.thresh           = sigmaW # THRSHLD_ORTHANT # 
    
    boundary_opt.lr               = 1e-7
    boundary_opt.cumul            = 0
    
    loss_dict = {}
    loss_dict['L'] = [0]
    
    #The Variables on which we optimize
    Variables = [p0,singular_points]
    
    #The optimizer in the interior of the orthants
    optimizer = torch.optim.LBFGS( Variables, max_eval=max_eval, lr = lr , 
                                   tolerance_grad = gtol, tolerance_change = tol,
                                   line_search_fn='strong_wolfe')
                                                        
    start = time.time()
    print('performing optimization...')
    boundary_opt.nit = -1
    
    ToSegments = SingularToSegments().apply
    
    def TestBoundary():
        """
        
        A test to see whether we should go to the boundary and change of orthant.
        
        """
        print("Check if we have to go to the boundary of ", boundary_opt.list_available)
        p0_tmp    = p0.clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
        nodes_tmp = singular_points.clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
        
        optimizer_tmp = torch.optim.SGD([p0_tmp,nodes_tmp], lr = boundary_opt.lr)
        
        tmp_current  = boundary_opt.list_test[-1]
        
        def closure_tmp():

            extremities, connections = ToSegments(nodes_tmp, singular_connections, 
                                                  mask_topo, mask_segments, ind_current_topo, 
                                                  boundary_opt.list_test, boundary_opt.list_available) 
            points, connections_points = Points_threshold_from_singular(extremities, connections, n_interp=n_interp)
            q0                         = Control_from_singular(extremities, connections, n_leaves, n_interp=n_interp)

            optimizer_tmp.zero_grad()
                
            L = loss(points, Variables[0], q0, connections_points)            
            L.backward(retain_graph=True)
                        
            return L
        
        avail_orth = torch.tensor(boundary_opt.list_test)
        if boundary_opt.list_available != []:
            print("GO TO BOUNDARY OF : ", boundary_opt.list_available)
            GoToBoundaryWithMomenta(p0_tmp, nodes_tmp, singular_connections,
                                    avail_orth, mask_topo, mask_segments)
        else:
            GoToBoundaryWithMomenta(p0_tmp, nodes_tmp, singular_connections, 
                                    avail_orth, mask_topo, mask_segments)
        
        optimizer_tmp.step(closure_tmp)
        
        selected_topo = boundary_opt.list_test[-1]
        
        if selected_topo == tmp_current:
            print("We should stay in the orhtant ", tmp_current)
        else:
            print("We can go to the boundary (old: {0} a,d new: {1})".format(tmp_current,selected_topo))
        
        return selected_topo, p0_tmp, nodes_tmp

    #Closure inside the orthant
    def closure_orthant():

        nonlocal ind_current_topo

        boundary_opt.nit += 1; it = boundary_opt.nit
        print("Iteration ",it)

        extremities, connections = ToSegments(singular_points, singular_connections, 
                                              mask_topo, mask_segments, ind_current_topo, 
                                              boundary_opt.list_test, boundary_opt.list_available) 
        points, connections_points = Points_threshold_from_singular(extremities, connections, n_interp=n_interp)
        q0                         = Control_from_singular(extremities, connections, n_leaves, n_interp=n_interp)

        optimizer.zero_grad()
            
        L = loss(points, Variables[0], q0, connections_points)

        if(folder2save != ''):
            if(boundary_opt.nit % 5 == 0) or export:
                loss_dict['L'].append(float(L.detach().cpu().numpy()))
                filesavename = "Iteration_"+str(boundary_opt.nit)
                template_labels = np.ones(points.shape[0])
                export_labeled_vtk(points.detach().cpu().numpy(),
                                   connections_points.detach().cpu().numpy(),
                                   template_labels,filesavename,folder2save)
                export_points(q0.detach().cpu().numpy(),"Q0_Iteration_"+str(boundary_opt.nit), folder2save)
                x, p, q = Flow(points, Variables[0], q0, Kv)
                export_labeled_vtk(x.detach().cpu().numpy(),
                                   connections_points.detach().cpu().numpy(),
                                   template_labels,'Deformed_'+filesavename,folder2save)


        L.backward(retain_graph=True)
        
        return L


    ############################### END OF THE CLOSURES DEFINITION, MAIN LOOP ###########################
    for i in range(maxiter):            # Fixed number of iterations
        
        if boundary_opt.nit <= maxiter:
            
            optimizer.step(closure_orthant)       # "Gradient descent" step in the interior.
            
            """res = Verification_LDDMM(loss, Variables[0], ind_current_topo, mask_topo, mask_segments, 
                                     singular_points, singular_connections, boundary_opt.list_test,
                                     boundary_opt.list_available, n_leaves)"""           
     
            previous_topo    = ind_current_topo
            ind_current_topo = boundary_opt.list_test[-1] 

            list_topo_close_enough, residual_dists = SelectAvailableTopo(singular_points, singular_connections, ind_current_topo, 
                                                                    dictionnary_topology_comparison,
                                                                    mask_topo_comparison, mask_topo, mask_segments,
                                                                    threshold = boundary_opt.thresh)

            boundary_opt.list_available  =  list_topo_close_enough.tolist()

            print("Residual ditances : \n", residual_dists)

            go_to_boundary = False

            if boundary_opt.allow_projection and len(boundary_opt.list_available) > 1:
                #Then we can go to the boundary contiguous to these available topologies
                go_to_boundary = True

            else:
                #We mus stay in the current orthant
                boundary_opt.list_available = [ind_current_topo]

            if not boundary_opt.allow_projection:
                for i in boundary_opt.last_topo_proj:
                    if residual_dists[i]!=0:
                        print("WE WENT FAR ENOUGH, allowing projections for the future iterations")
                        boundary_opt.allow_projection = True
                        break
                
            print(boundary_opt.list_test, boundary_opt.allow_projection, boundary_opt.list_available)
            

            if go_to_boundary:
                
                selected_topo, new_p0, new_vertices = TestBoundary()
                
                if selected_topo != ind_current_topo:
                    Current_topo     = mask_topo[:,selected_topo]
                    Previous_topo    = mask_topo[:,ind_current_topo]
                    Current_segments = Current_segments  = singular_connections[mask_segments[:,selected_topo]==1,:]
                    p0.data = new_p0.data
                    singular_points.data = new_vertices.data
                    p0 = RearangeMomenta(p0, singular_points, singular_connections,
                                            ind_current_topo, selected_topo,
                                            boundary_opt.list_available, mask_topo, 
                                            mask_segments, n_interp, n_leaves)

                    for i in range(mask_topo.shape[0])[1:]:
                        
                        test_common = (mask_topo[i,boundary_opt.list_available]==1).all()
                                
                        if Current_topo[i] and not test_common:
                            #Then this point is associated to a new branch that will grow
                            if Previous_topo[i]==1:
                                #Still shared with previous topo
                                pos_current  = torch.where(Current_segments[:,1]==i)[0]
                                p0[pos_current*(n_interp-2):(pos_current+1)*(n_interp-2),:].data *= 000000.1
                                
                    ind_previous_topo = ind_current_topo
                    ind_current_topo  = selected_topo

                    #then reset the optimizer
                    print("RESET THE OPTIMIZER")
                    optimizer = torch.optim.LBFGS( Variables, max_eval=max_eval, lr=lr, 
                                       tolerance_grad = gtol, tolerance_change = tol,
                                       line_search_fn='strong_wolfe')
                    
                    boundary_opt.allow_projection = False
                    boundary_opt.cumul            = boundary_opt.nit + 70
                    boundary_opt.list_test        = [selected_topo]
                    boundary_opt.last_topo_proj   = boundary_opt.list_available
                    boundary_opt.list_available   = [ind_current_topo]
                    boundary_opt.went2boundary = True
                else: 
                    #do nothing, keep moving in the current orthant
                    boundary_opt.allow_projection = False
                    boundary_opt.cumul            = boundary_opt.nit + 70
                    boundary_opt.list_test        = [previous_topo,ind_current_topo]
                    boundary_opt.list_available   = [ind_current_topo]
                    
            else:
                boundary_opt.went2boundary = False
                if boundary_opt.cumul - boundary_opt.nit <= 0:
                    boundary_opt.allow_projection = True

            #We must update the position of the other points that were not used in the construction of the spatial tree
            current_topo = (mask_topo[:,ind_current_topo]).view(-1,1)
            for i in range(singular_points.shape[0]):
                if not mask_topo[i,ind_current_topo]: 
                    parent                             = FindParent(i, singular_connections, current_topo)
                    diff = (singular_points[parent,:]).data - (singular_points[i,:]).data
                    (singular_points[i,:]).data += diff
               
            print("Previous topology : ", previous_topo, "Current topology : ",  ind_current_topo)
        
        else:
            print("Maximum iteration reached ({0})".format(boundary_opt.nit))
            break
        
    total_time = round(time.time()-start,2)
    print('Optimization time : ',total_time,' seconds')

    if(folder2save != ''):
        try:
            os.mkdir(folder2save)
        except OSError:
            pass
        loss_dict['Time'] = total_time
        loss_dict['it'] = boundary_opt.nit
        with open(folder2save+'/dict_'+savename+'.pkl','wb') as f:
            pickle.dump(loss_dict,f)
            
    return (Variables[0], Variables[1], ind_current_topo, boundary_opt.nit, total_time)
    
    
