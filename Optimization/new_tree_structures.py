# -*- coding: utf-8 -*-
"""
Created on Wed May 26 19:23:04 2021
"""
from constants import THRSHLD_ORTHANT, THRSHLD_LENGTH, N_INTERP, THRSHLD_CONSTRUCTION

from registration import Hamiltonian_points

from keops_utils import TestCuda
import torch

use_cuda,torchdeviceId,torchdtype,KeOpsdeviceId,KeOpsdtype,KernelMethod = TestCuda()


def Points_threshold_from_singular(pts, connections, n_interp=N_INTERP):
    """
    Here connections input correspond to potentially too small segments. 
    Needs to be taken into account in this function.
    
    Parameters
    ----------
    @param : pts         : torch tensor
                           n-points x d-dimension points.
                 
    @param : connections : torch Long tensor
                           m-connections x 2-dim tensor containing pair of connected points' indices.

    @param : n_interp    : int
                           The number of points in the branch.

    Returns
    -------
    @output : V : Torch Float Tensor
                       Average euclidean distance between connected points in the tree.
    @output : L : Torch Long Tensor
                       Standard deviation of the euclidean distance between connected points
    """
        
    n_redundant = 0 #connections.shape[0] - connections[:,0].unique().shape[0]
    
    seen = []
    
    mask_ok = torch.ones(connections.shape[0])
    connections = connections.unique(dim=0)
    
    output = torch.unique(pts, dim=0)
      
    list_seg_too_small = []
    indices_avoided    = []
    ind_con2avoid      = []
    #Remove the segments too short 
    for i, con in enumerate(connections):
        if (pts[con[0],:]-pts[con[1],:]).norm() <= THRSHLD_LENGTH:
            ind_con2avoid.append(i)
            list_seg_too_small.append(con)
            indices_avoided += con
        
    seg_too_small = len(list_seg_too_small)
        
    """print(seg_too_small)
    print("List : ", list_seg_too_small)
    print("Connections : ")
    print(connections)"""
        
    #print("NUMBER OF SEGMENTS TOO SMALL : ", seg_too_small)
    
    n_V         = n_interp*(connections.shape[0] - seg_too_small) #+ pts.shape[0] - output.shape[0]
        
    V = torch.zeros((n_V, pts.shape[1])).to(dtype=torchdtype, device=torchdeviceId)
    
    n_pts_current = pts.shape[0] - len(indices_avoided)
    
    list_all_con = []
    
    cpt = 0
    
    new_con = connections.clone()
            
    for i,pt in enumerate(pts):
        if i not in indices_avoided:
            V[cpt,:] = pt
            for k,con in enumerate(connections):
                if con[0]==i and con[0] not in indices_avoided:
                    connections[k,0] = cpt
                if con[1]==i and con[1] not in indices_avoided:
                    connections[k,1] = cpt
            cpt+=1
                       
    if cpt != n_pts_current:
        print("Wrong count in Points_threshold_from_singular")
    #V[:pts.shape[0],:] = pts
    
    #print("V shape : ", V.shape)
    
    for ind_test,con in enumerate(connections): #[c for c in connections]:
        
        if ind_test not in ind_con2avoid:
            start = V[con[0],:]
            end   = V[con[1],:]

            if (end-start).norm() >= THRSHLD_LENGTH: 
                
                #Compute the points as barycenters of the extremities
                for j in range(V.shape[1]):
                    V[n_pts_current:n_pts_current+n_interp-2,j] = start[j]+(end[j]-start[j])*(torch.linspace(0, 1, steps = n_interp)[1:-1]).to(dtype=torchdtype, device=torchdeviceId)
                
                #Add the connections
                interp_connections      = torch.zeros((n_interp-1,2)).to(dtype=torch.long)
                interp_connections[:,0] = torch.arange(0,n_interp-1)
                interp_connections[:,1] = torch.arange(1,n_interp)
            
                interp_connections        += n_pts_current-1
                interp_connections[0,0]   = con[0]
                interp_connections[-1,1]  = con[1]
                
                list_all_con.append(interp_connections)
                
                for i in range(interp_connections.shape[0]):
                    if interp_connections[i,0]== interp_connections[i,1] :
                        print("GOT THE PROBLEM ! In the connections : ", interp_connections[i,0], interp_connections[i,1])
                        print(con)
                    if (V[interp_connections[i,0],:]==V[interp_connections[i,1],:]).all():
                        print("GOT THE PROBLEM ! In the points : ", V[interp_connections[i,0],:], interp_connections[i,0], interp_connections[i,1])
                        print("For segment : ", con, " and pts : ", start, end)
                        
                n_pts_current += n_interp-2

    L = torch.cat(list_all_con, 0).to(dtype=torch.long, device=torchdeviceId)
    
    return V.contiguous(), L.contiguous()
    
    
def Control_from_singular(pts, connections, n_leaves, n_interp=N_INTERP):
    """
    Compute the control points from the active segments and the current topology. 
    
    If we are close to a orthant's boundary, we just have to fill the positions 
    of the inactivted segments with new control points. 
    
    If we are at a boundary, the control points will have 
    (n_interp \times n_shrunk_seg) points overlapping.
    
    
    Parameters
    ----------
    @param : pts         : torch tensor
                           n-points x d-dimension points.
                 
    @param : connections : torch Long tensor
                           m-connections x 2-dim tensor containing pair of connected points' indices.

    @param : n_interp    : int
                           The number of points in the branch.

    Returns
    -------
    @output : V : Torch Float Tensor
                       Average euclidean distance between connected points in the tree.
    @output : L : Torch Long Tensor
                       Standard deviation of the euclidean distance between connected points
    """
    
    n_branches  = 2*n_leaves-3
    n_V         = (n_interp-2)*n_branches #Remove redundant points, first point of each branch that is not the root
        
    V = torch.zeros((n_V, pts.shape[1])).to(dtype=torchdtype, device=torchdeviceId)
    
    #print("V SHAPE : ", V.shape)
    
    n_pts_current = 0 #pts.shape[0]
    
    list_all_con = []
    
    #V[:pts.shape[0],:] = pts

    #print(connections.shape, n_branches)

    for con in connections:
        
        start = pts[con[0],:]
        end   = pts[con[1],:]
        if (end-start).norm() >= THRSHLD_LENGTH: 
            for i in range(V.shape[1]):
                V[n_pts_current:n_pts_current+n_interp-2,i] = start[i]+(end[i]-start[i])*(torch.linspace(0, 1, steps = n_interp)[1:-1]).to(dtype=torchdtype, device=torchdeviceId)
        else:
            V[n_pts_current:n_pts_current+n_interp-2,:] = start
            
        n_pts_current += n_interp-2

    #print(V)

    return V.contiguous()
    
    
def Points_from_singular(pts, connections, n_interp=N_INTERP):
    """
    
    Parameters
    ----------
    @param : pts         : torch tensor
                           n-points x d-dimension points.
                 
    @param : connections : torch Long tensor
                           m-connections x 2-dim tensor containing pair of connected points' indices.

    @param : n_interp    : int
                           The number of points in the branch.

    Returns
    -------
    @output : V : Torch Float Tensor
                       Average euclidean distance between connected points in the tree.
    @output : L : Torch Long Tensor
                       Standard deviation of the euclidean distance between connected points
    """
        
    n_redundant = 0 #connections.shape[0] - connections[:,0].unique().shape[0]
    
    seen = []
    
    #Count the number of redundant points in the connections
    for i in connections[:,0]:
        if i in connections[:,1] or i in seen:
            n_redundant+=1
        seen.append(i)
    
    print("N redondants : ", n_redundant)
    
    n_V         = n_interp*connections.shape[0] - n_redundant
        
    V = torch.zeros((n_V, pts.shape[1])).to(dtype=torchdtype, device=torchdeviceId)
    
    n_pts_current = pts.shape[0]
    
    list_all_con = []
    
    V[:pts.shape[0],:] = pts
    
    for j, con in enumerate(connections):
        
        start = pts[con[0],:]
        end   = pts[con[1],:]
        
        if (start-end).norm(p=2) >= THRSHLD_LENGTH: #(start.data!=end.data).any(): #
        
            for i in range(V.shape[1]):
                V[n_pts_current:n_pts_current+n_interp-2,i] = start[i]+(end[i]-start[i])*(torch.linspace(0, 1, steps = n_interp)[1:-1]).to(dtype=torchdtype, device=torchdeviceId)
            
            interp_cons = torch.zeros((n_interp-1,2))
            interp_cons[:,0] = torch.arange(0,n_interp-1)
            interp_cons[:,1] = torch.arange(1,n_interp)
        
            interp_cons += n_pts_current-1
            interp_cons[0,0]  = con[0]
            interp_cons[-1,-1] = con[1]
            
            list_all_con.append(interp_cons)
        
        n_pts_current += n_interp-2

    if list_all_con == []:
        for j, con in enumerate(connections):
            start = pts[con[0],:]
            end   = pts[con[1],:]
            print(end-start)
    L = torch.cat(list_all_con, 0).to(dtype=torch.long, device=torchdeviceId)
    
    #print(V.shape)
    
    return V.contiguous(), L.contiguous()
        

def SelectAvailableTopo(pts, connections, ind_current_topology, dictionnary_topo_comparison,
                        mask_topology_comparison, points_activation_mask, segments_activation_mask, 
                        threshold = THRSHLD_ORTHANT):
    """
    
    This is not the best way to code it... In fact we need the active segments and not the points...
    
    
    Parameters
    ----------
    @param : pts                     : torch tensor
                                       n-points x d-dimension points.
                                       The singular points, segments extremities. 
                 
    @param : connections             : torch Long tensor
                                       m-connections x 2-dim tensor containing pair of connected points' indices.
                                       Connections between the singular points

    @param : ind_current_topology    : int
                                       XXX

    Returns
    -------
    @output : selected_topologies : Torch Boolean Tensor
                                    Mask of the selected topologies regarding the distance to the current orthant's borders.

    """
    n_topo = segments_activation_mask.shape[1]

    if ind_current_topology == -1:
        return torch.tensor([k for k in range(n_topo)]).to(torch.long).to(device=torchdeviceId), torch.zeros((n_topo)).to(device=torchdeviceId)
    else:
        # get all columns with at most 2 values of difference with current topology
        #
        # WARNING : All the topologies of activated segments correspond to binary trees (so at 
        #           most 2*n_leaves - 3 segments), but to be able to compare we can either store all the 
        #           possible segments (with the merges) or we merge afterward.
        #
            
        segments_of_interest = mask_topology_comparison[:,ind_current_topology]

        compare = (mask_topology_comparison.to(dtype=torchdtype).add(-segments_of_interest.to(dtype=torchdtype).view(-1,1))).abs()

        #######################################
        current_topo = segments_activation_mask[:,ind_current_topology]
        compare = torch.zeros(segments_activation_mask.shape).to(device=torchdeviceId)
        
        compare_topos = torch.zeros(segments_activation_mask.shape).to(device=torchdeviceId)
        
        distances = torch.zeros((n_topo)).to(device=torchdeviceId)
        
        distances_non_null = torch.zeros((n_topo)).to(device=torchdeviceId)
        
        for i,test in enumerate(current_topo):
        
            if test: #this segment exists in the current topo
                
                con = connections[i,:]
                d2 = (pts[con[1],:]-pts[con[0],:]).norm(p=2).square()
                
                for j in range(n_topo):
                    
                    if not segments_activation_mask[i,j] and not points_activation_mask[con[1],j] and con[1] in connections[:,0]: # 
                    #then the segment has to shrink if we want to go to this topology
                        
                        compare_topos[i,j] = 1
                        
                        if d2.sqrt() >= threshold:
                            compare[i,j] = 1
                            distances_non_null[j] += d2
                        
                        distances[j] += d2
                            
        distances = distances.sqrt()
        distances_non_null = distances_non_null.sqrt()
        #######################################

        #get the lengths of the segments
        V0, V1 = pts.index_select(0,connections[:,0]), pts.index_select(0,connections[:,1])
        u                                    =    (V1-V0)
        lengths                              =    (u**2).sum(1)[:, None].sqrt()
                
        # Then check whether the active component of current topology is <= threshold
        lengths_of_interest = lengths.view(-1,1) * compare_topos
      
        #print(torch.where(compare.sum(dim=0)<=0)[0])
        #print(torch.where((lengths_of_interest-threshold<=0).all(0))[0])
      
        #print((lengths_of_interest-threshold).sum(dim=0))
      
        #Norm inf:
        #selected_topologies_projected = torch.where((lengths_of_interest-threshold<=0).all(0))[0] #first selection
        #selected_topologies_projected = torch.where(compare.sum(dim=0)<=0)[0]
        
        #Norm 2:
        #selected_topologies_projected = torch.where((lengths_of_interest.square().sum(dim=0).sqrt()-threshold).sum(dim=0)<=0)[0] # the overall distance should be smaller than the threshold

        #Length:
        selected_topologies_projected = torch.where((lengths_of_interest.sum(dim=0)-threshold)<=0)[0] # the overall distance should be smaller than the threshold

        contiguous_orthants = dictionnary_topo_comparison[int(ind_current_topology)]
        
        selected_topologies = [] 
        
        for i in selected_topologies_projected:
            if True: # int(i) in contiguous_orthants: #The commented criterion corresponds to the case of a limited number of changes between the orthants
                selected_topologies.append(int(i))
        
        selected_topologies = torch.tensor(selected_topologies)
        
        selected_topologies.to(device=torchdeviceId)

        return selected_topologies, distances_non_null


def GoToBoundary(pts, connections, indices_available_topologies, mask_points, mask_segments):
    """
    
    Given a set of possible topologies, finds the common border between the orthants and get to 
    this border.
    
    """
    n_pts = pts.shape[0]
        
    for i in range(n_pts):
        
        test = (mask_points[i,indices_available_topologies.to(dtype=torch.long)]==0).any()
        #test = (mask_points[i,ind_current_topo]==1 and mask_points[i,ind_new_topo]==0)
        
        if test: # i not in corresp_final.keys(): # 
            parent          = FindParent(i, connections, mask_points[:,indices_available_topologies.to(dtype=torch.long)], mask_segments)
            diff = (pts[parent,:]).data - (pts[i,:]).data
            (pts[i,:]).data += diff
    
    return 
    
    
    
def GoToBoundaryWithMomenta(momenta, singular_points, singular_connections, indices_available_topologies, mask_points, mask_segments):
    """
    
    Given a set of possible topologies, finds the common border between the orthants and get to 
    this border.
    
    """
    n_pts = singular_points.shape[0]
    
    for i in range(n_pts):
        
        #If a point is not activated in one topology at the boundary, must skrink the branch
        test = (mask_points[i,indices_available_topologies.to(dtype=torch.long)]==0).any()
        
        if test: 
            parent          = FindParent(i, singular_connections, mask_points[:,indices_available_topologies.to(dtype=torch.long)], mask_segments)
            diff = (singular_points[parent,:]).data - (singular_points[i,:]).data
            (singular_points[i,:]).data += diff
                
    
    return momenta, singular_points    



def RearangeMomenta(momenta, singular_points, singular_connections, ind_previous_topo, ind_new_topo, indices_available_topologies, mask_points, mask_segments, n_interp, n_leaves):
    """
    
    Given a set of possible topologies, finds the common border between the orthants and get to 
    this border.
    
    """
    n_pts      = singular_points.shape[0]
    n_branches = 2*n_leaves-3
    n_ctrl_pts = (n_interp-2)*n_branches
    
    Current_topo  = mask_points[:,ind_new_topo]
    Previous_topo = mask_points[:,ind_previous_topo]

    dict_parents = {}

    
    for i in range(n_pts):
        
        test = (mask_points[i,indices_available_topologies]==0).any()
        #test = (mask_points[i,ind_current_topo]==1 and mask_points[i,ind_new_topo]==0)
        
        if test: # and Current_topo[i]: # i not in corresp_final.keys(): # 
            parent          = FindParent(i, singular_connections, mask_points[:,indices_available_topologies], mask_segments)

            if parent not in dict_parents.keys():
                dict_parents[parent] = [i]
            else:
                dict_parents[parent].append(i)

    Current_segments  = singular_connections[mask_segments[:,ind_new_topo]==1,:]
    Previous_segments = singular_connections[mask_segments[:,ind_previous_topo]==1,:]
    
    reorder_ind = torch.zeros((2*n_leaves-3,1),dtype=torch.long, device=torchdeviceId)
    
    print(Current_segments)
    print(Previous_segments)
    
    for i in range(n_pts)[1:]:
        
        test_common = (mask_points[i,indices_available_topologies]==1).all()
        
        if Current_topo[i] and test_common:
            #Then it is a shared point
            
            pos_previous = torch.where(Previous_segments[:,1]==i)[0]
            pos_current  = torch.where(Current_segments[:,1]==i)[0]
            
            print(pos_current)
            print(pos_previous)
            
            reorder_ind[pos_current] = pos_previous
                        
        if Current_topo[i] and not test_common:
            #Then this point is associated to a new branch that will grow
            
            if Previous_topo[i]==1:
                #Still shared with previous topo
                pos_previous = torch.where(Previous_segments[:,1]==i)[0]
                pos_current  = torch.where(Current_segments[:,1]==i)[0]
                
                print(pos_current)
                print(pos_previous)
                
                print("TEST_WEIGHT MOMENTA")
                momenta[pos_previous*(n_interp-2):(pos_previous+1)*(n_interp-2),:].data *= 000000.1
                
                reorder_ind[pos_current] = pos_previous

                for parent in dict_parents.keys():
                    if i in dict_parents[parent]:
                        dict_parents[parent].remove(i)
                        break
                        
    #We erased the shared points with the previous topo from dict_parent, now we find 
    #the momenta associated to growing branches from the parents
    
    remaining_inds = [k for k in range( Current_segments.shape[0])]
    
    for ind in reorder_ind:
        if ind in remaining_inds:
            remaining_inds.remove(ind)
    
    for i in range(n_pts)[1:]:
        test_common = (mask_points[i,indices_available_topologies]==1).all()
        
        if Current_topo[i] and not test_common and Previous_topo[i]==0:
            
            print(i, Current_topo[i], test_common, Previous_topo[i])
            
            parent = FindParent(i, singular_connections, mask_points[:,indices_available_topologies], mask_segments)
            
            print("Paretn i : ", parent)
            
            dict_parents[parent]    
            other_ind = -1
            for j,ind in enumerate(dict_parents[parent]):
                if Previous_topo[ind]==1:
                    other_ind = int(dict_parents[parent].pop(j))
                    print(other_ind)
                    break
            
            if other_ind==-1:
                print("UH OH")
                print(dict_parents[parent])
                print(Previous_topo[dict_parents[parent]])
                
                other_ind = remaining_inds.pop(0)
                print(other_ind)
                print("Parent other ind : ", FindParent(other_ind, singular_connections, mask_points[:,indices_available_topologies], mask_segments))
                    
            
            print(i)
            print("check where : ", torch.where(Current_segments[:,1]==i))
            pos_current  = torch.where(Current_segments[:,1]==i)[0]
            pos_previous = torch.where(Previous_segments[:,1]==other_ind)[0]
            
            reorder_ind[pos_current] = pos_previous
                
            print(pos_current)
            print(pos_previous)
    print(reorder_ind)

    ordered_momenta = momenta.clone()

    #Now we re-order the momenta so that we get the correct positions
    for i,previous_i in enumerate(reorder_ind):
        
        if i != previous_i:
            
            ordered_momenta[i*(n_interp-2):(i+1)*(n_interp-2),:].data = momenta[previous_i*(n_interp-2):(previous_i+1)*(n_interp-2),:].data
        
    momenta.data = ordered_momenta.data
        
    return momenta


def ComputeResidual(pts, connections, ind_current_topology, available_topologies, 
                    points_activation_mask, segments_activation_mask):
    """
    For each other available, compute the length to go through if it were selected
    """
    
    current_topo = segments_activation_mask[:,ind_current_topology]
    
    distances = torch.zeros((len(available_topologies))).to(device=torchdeviceId)
    
    for i,test in enumerate(current_topo):
        
        if test: #this segment exists in the current topo
            
            con = connections[i,:]
            d2 = (pts[con[1],:]-pts[con[0],:]).norm().square()
            
            for j,ind in enumerate(available_topologies):
                
                if not segments_activation_mask[i,ind] and not points_activation_mask[con[1],ind]:
                #then the segment has to shrink if we want to go to this topology
                    
                    distances[j] += d2
                    
    distances.sqrt()
    
    return distances
                            


def FindParent(child, connections, mask_points, mask_segments = None):

    found = False
    to_return = 0
    indices_lines = torch.where(connections[:,1]==child)
    
    #print(connections,child)
    
    #print(indices_lines[0])
    
    parent        = connections[indices_lines[0],0]

    for i in indices_lines[0]:
        #print(i)
        ind = int(i)
        #print("First test : ", mask_points[connections[ind,0],:], "for ", connections[ind,0])
        if (mask_points[connections[ind,0],:]==1).all():
            found = True
            to_return = int(connections[ind,0])
            break
            #return int(connections[ind,0])
            
    #None of the potential parents is active, must recursively find the parent
    #print("None of the potential parents is active, must recursively find the parent")
    #print("Current child : ", child, ", indices_lines : ", indices_lines)
    if not found:
        for i in indices_lines[0]:
            #print("Parent : ", i)
            ind = int(i)
            ancestor = FindParent(connections[ind,0], connections, mask_points, mask_segments = mask_segments)
            #print("Second test : ", mask_points[ancestor,:], "for ", ancestor)
            if (mask_points[ancestor,:]==1).all():
                found = True
                to_return = ancestor
                break
                #return ancestor

    if to_return==0:
        print("NO PARENT FOUND........ !!!!!!!!!!!!!! \n")
    return to_return


def MergeSegments(connections, activated_segments):
    """
    
    Select the segment that can be used in the data attachment. 
    For now, a segment [a, b] is inactivated if :
    - b is inactive.
    - a is inactive and b is the end of another segment. 
    
    """
    
    new_connections        = connections.new(connections)
    new_activated_segments = torch.clone(activated_segments)

    active_connections = torch.masked_select(connections, activated_segments.view(-1,1)).view(-1,2)

    indices = active_connections.unique()

    for i in indices:

        m_start = ((connections[:,0]==i) * activated_segments ).to(dtype=torch.bool)
        m_end   = ((connections[:,1]==i) * activated_segments ).to(dtype=torch.bool) 

        #we deactivate if there is another segment ending with con[1]
        if torch.count_nonzero(m_start)==1 and torch.count_nonzero(m_end)==1: 

            print("There will be a merge at : ", i)

            s1 = torch.where(m_start==1,m_start, False)
            s2 = torch.where(m_end==1, m_end, False)

            new_activated_segments[s1] = 0
            new_activated_segments[s2] = 0
                        
            #new_seg    = torch.tensor([connections[s2,0],connections[s1,1]]).to(dtype=torch.long, device=torchdeviceId)
            #activation = torch.tensor([1]).to(dtype=torch.bool, device=torchdeviceId)
            
            #new_connections        = torch.cat((new_connections,new_seg.view(-1,2)), 0)
            #new_activated_segments = torch.cat((new_activated_segments,activation), 0)

    return new_activated_segments.contiguous(), new_connections.contiguous()


def ActivateSegments(pts, connections, activated_points):
    """
    
    Select the segment that can be used in the data attachment. 
    For now, a segment [a, b] is inactivated if :
    - b is inactive.
    - a is inactive and b is the end of another segment. 
    
    """
    
    activated_segments = torch.ones(connections.shape[0]).to(dtype=torch.bool, device=torchdeviceId)

    for i, con in enumerate(connections):
        if not activated_points[con[0]] and not activated_points[con[1]]: #both ends are inactive
            activated_segments[i] = 0
            
        elif not activated_points[con[0]]: 
            
            if torch.count_nonzero(connections[:,1]==con[1])>1: #we deactivate if there is another segment ending with con[1]
                activated_segments[i] = 0
    
        #elif not activated_points[con[1]] or not activated_points[con[0]]: 
        #    activated_segments[i] = 0
    
    return activated_segments
        
        
def StoreIndBackward(dictionnary, connections, ind_current_topo, topo_to_check,
                     points_activation_mask, segments_activation_mask, 
                     ind_start, ind_end, ref_start, ref_end):
    """
    Stores for the backward the indices of interest. 
    The idea is that every points acitve in the topologies should receive the gradient of 
        - The corresponding created point in the tree that will be used in the data attachment term,
        - If it is not activated in the current topology, the gradient of the points sharing a segment's extremity. 
            (Ex: point 6 is not acivated in the current topology but not in another one. The segment [2,4] exists
            in the current topology and [6,4] doesn't, then the gradient of 2 for the segment [2,4] will be also 
            attributed to 6 in the other topology.
    
    """
    
    #print("\n\n\n\n\n\n New topo : ", topo_to_check)
    
    
    for ind_con in range(connections.shape[0]):
                            
        if segments_activation_mask[ind_con,topo_to_check]==1: 
        
            a = int(connections[ind_con,0])
            b = int(connections[ind_con,1])
            
            ind_diff = ind_start
            #print(ind_start, ind_end, connections[ind_con,:], ind_con)
            
            if ind_start in connections[ind_con,:]:
        
                #Add a reference in the dictionnary
                if ind_start not in dictionnary.keys():
                    dictionnary[ind_start] = []
                    
                if ref_start not in dictionnary[ind_start]:
                    dictionnary[ind_start].append(ref_start)
                
                """if a != ind_start:
                    ind_diff = a
                else:
                    ind_diff = b
                                
                #Then we found a point that is not activated in the current topology 
                #and we need to attribute it a gradient
                if (points_activation_mask[ind_diff,ind_current_topo] == 0 and 
                   points_activation_mask[ind_diff,topo_to_check] == 1 and ind_diff == b):
                                              
                    if ind_diff not in dictionnary.keys():
                        dictionnary[ind_diff] = []
                    
                    if ref_start not in dictionnary[ind_diff]:
                        dictionnary[ind_diff].append(ref_start)
                        print("Adding from [{0}-{1}] the point {2} for the grad of {3}".format(ind_start,ind_end,ref_start,ind_diff)) 
                
                elif points_activation_mask[ind_diff,topo_to_check] == 0:
                    print("\n\n\n\n\n WARNING Here a point is selected in a connection, but not in the points \n\n\n\n")
                """
            if ind_end in connections[ind_con,:]:
        
                if ind_end not in dictionnary.keys():
                    dictionnary[ind_end]   = []
        
                if ref_end not in dictionnary[ind_end]:
                    dictionnary[ind_end].append(ref_end)
    
                if segments_activation_mask[ind_con,ind_current_topo]==0:
                    
                    ind_diff = -1
                    
                    if a != ind_end:
                        ind_diff = a
                    else:
                        ind_diff = b
                    
                    #Then we found a point that is not activated in the current topology 
                    #and we need to attribute it a gradient
                    if (points_activation_mask[a,ind_current_topo] == 0 and 
                       points_activation_mask[a,topo_to_check] == 1 and ind_diff == a):
                        if a not in dictionnary.keys():
                            dictionnary[a] = []
                        
                        if ref_start not in dictionnary[a]:
                            dictionnary[a].append(ref_start)
                            
    return
        
        
        
        
class SingularToSegments(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    
    In the forward pass we receive a Tensor containing the input and return
    a Tensor containing the output. self is a context object that can be used
    to stash information for backward computation. You can cache arbitrary
    objects for use in the backward pass using the ctx.save_for_backward method.
    """
    def __init__(self):
        super(SingularToSegments, self).__init__()
        self.ind_selected_topo = -1
        
    def update_topo(self, ind):
        self.ind_selected_topo = ind

    def get_topo(self):
        return self.ind_selected_topo

    @staticmethod
    def forward(self, pts, connections, 
                mask_topo, segments_activation_mask, 
                ind_current_topo, list_previous = [], 
                available_topologies = []):
        """
        
        In this forward function the output could be : 
        - the selected segments extremities
        - the whole points and connectivities... to be decided
        
        Parameters
        ----------
        @param : V : torch tensor
                     n-points x d-dimension points.
                     
        @param : F : torch Long tensor
                     m-connections x 2-dim tensor containing pair of connected points' indices.

        @param : n_interp : int
                     The number of points in the branch.

        Returns
        -------
        @output : V : Torch Float Tensor
                           Average euclidean distance between connected points in the tree.
        @output : L : Torch Long Tensor
                           Standard deviation of the euclidean distance between connected points
        """

        n_seg = 0
        dim = pts.shape[1]
        list_seg = [] #List containing the unique segments 

        if available_topologies == []:
            available_topologies = [ind_current_topo]
            
        n_topo = len(available_topologies)
                
        if ind_current_topo != -1:
            #select the correct activated segments
            activated_segments = segments_activation_mask[:,ind_current_topo]             
        else:
            activated_segments = torch.ones(connections.shape[0]).to(dtype=torchdtype, device=torchdeviceId)

        correspondences_per_topo = [{} for i in range(n_topo)]

        for i, con in enumerate(connections):
            
            start = pts[con[0],:]
            end   = pts[con[1],:]
            
            ind_start = int(con[0].detach().cpu())
            ind_end   = int(con[1].detach().cpu())

            if activated_segments[i]:
                  
                #Then create the segment and check if it already exists
                segment = torch.Tensor(2, pts.shape[1]).to(dtype=torchdtype, device=torchdeviceId)
                torch.cat([start,end], out=segment)
                
                """l_bool = [(segment==s).all() for s in list_seg if s.shape==segment.shape]
                
                if any( l_bool ):
                    ind_segment = l_bool.index(True)
                    ref_start = 2*ind_segment
                    ref_end   = 2*ind_segment+1
                else:"""
                list_seg.append(segment)
                ref_start = 2*n_seg
                ref_end   = 2*n_seg+1
                
                #for the gradient projection, stores the connected points
                for i,topo in enumerate(available_topologies):
                    
                    dict_correspondences = correspondences_per_topo[i]
                    
                    StoreIndBackward(dict_correspondences, connections, ind_current_topo, topo,
                                     mask_topo, segments_activation_mask, 
                                     ind_start, ind_end, ref_start, ref_end)
                     
                #if True:
                n_seg+=1

        print("N seg : ", n_seg)
        V = torch.zeros((2*n_seg, dim)).to(dtype=torchdtype, device=torchdeviceId)
        L = torch.zeros((n_seg, 2)).to(dtype=torch.long, device=torchdeviceId)

        for dict_correspondences in correspondences_per_topo:
            for k in dict_correspondences.keys():
                dict_correspondences[k] = torch.Tensor(dict_correspondences[k]).to(dtype=torch.long, device=torchdeviceId)
                        
        for i, seg in enumerate(list_seg):
                             
            V[2*i,:]   = seg[:dim]
            V[2*i+1,:] = seg[dim:]
            
            L[i,0] = 2*i
            L[i,1] = 2*i+1

        self.all_correspondences             = correspondences_per_topo
        self.ind_current_topo                = ind_current_topo
        self.list_previous                   = list_previous
        self.available_topologies            = available_topologies
        self.save_for_backward(pts)

        return V, L

        
    @staticmethod
    def backward(self, grad_points, grad_connections):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        
        Here we need in ctx : the association between input[i,:] and the different output[j,:]
        """

        pts = self.saved_tensors[0]     
        n_pts, dim = pts.shape[0], pts.shape[1]
        
        ind_current_topo         = self.ind_current_topo                 
        all_correspondences      = self.all_correspondences
                
        n_topo_available     = len(self.available_topologies)
        available_topologies = torch.tensor(self.available_topologies).to(device=torchdeviceId)

        grad_input = torch.zeros(pts.shape).to(dtype=torchdtype, device=torchdeviceId)
          
        selected_gradients = torch.zeros((n_pts,dim,n_topo_available)).to(dtype=torchdtype, device=torchdeviceId)
          
        list_previous_topo = self.list_previous
                  
        for ind_topo,corresp in enumerate(all_correspondences):
            #Retrieve the gradient with respect to the template's vertices
            for k in corresp.keys():
                selected_gradients[k,:,ind_topo] = grad_points.index_select(0,corresp[k]).sum(dim=0) #/len(corresp[k])

        print("topo available : ", available_topologies)

        if n_topo_available > 1:
            
            #### Project the gradient onto different orthants ###
            projection_matrix_norm = selected_gradients.norm(dim=[0,1])
            
            print("gradient norms : ", projection_matrix_norm)
            
            #Select the projection maximizing the norm of the projected gradient
            selected_topo = projection_matrix_norm.argmax()
            grad_input = selected_gradients[:,:,selected_topo]
            
            final_ind = available_topologies[selected_topo]
            
        else:
            final_ind     = ind_current_topo
            print(grad_input.shape)
            grad_input    = selected_gradients[:,:,0]
            
        if len(list_previous_topo)>1: #means that we stored two previous topologies
            list_previous_topo.pop(0)
        list_previous_topo.append(int(final_ind))

        return grad_input, None, None, None, None, None, None
