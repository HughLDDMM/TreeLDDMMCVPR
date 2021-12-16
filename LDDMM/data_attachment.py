#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 11:30:25 2019
"""

import os
import sys
#sys.path.append(os.path.abspath("../OT"))

import torch
from torch.autograd import Variable
import numpy as np


from keops_utils import OrientedGaussLinKernel
from keops_utils import TestCuda

# Cuda management
use_cuda,torchdeviceId,torchdtype,KeOpsdeviceId,KeOpsdtype,KernelMethod = TestCuda(verbose=False)


####################################################################
# Computing the input structures
# ^^^^^^^^^^^^^^^^^^^^


def Compute_structures_surface(V,F):
    """
    For the linear normal kernel
    
    Parameters
    ----------
    @param : V : torch tensor
                 n-points x d-dimension points.
    @param : F : torch Long tensor
                 m-connections x 2-dim tensor containing pair of connected points' indices.

    Returns
    -------
    @output : centers        : torch tensor
                               npoints-1 x d-dimension points, the centers of each face.
    @output : length         : float
                               npoints-1 x 1-dimension tensor, the areas of each face.
    @output : normals        : torch tensor
                               npoints-1 x d-dimension normalized vectors of normals of each face. 
    """

    V0, V1, V2 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1]), V.index_select(0, F[:, 2])   
    centers, normals =  (V0 + V1 + V2) / 3, .5 * torch.cross(V1 - V0, V2 - V0)    
    length = (normals ** 2).sum(dim=1)[:, None].sqrt()
    
    return centers, length, normals/(length+1e-5)


####################################################################
# Computing the input structures
# ^^^^^^^^^^^^^^^^^^^^
def Compute_lengths(V,F):
    """
    Parameters
    ----------
    @param : V : torch tensor
                 n-points x d-dimension points.
    @param : F : torch Long tensor
                 m-connections x 2-dim tensor containing pair of connected points' indices.

    Returns
    -------
    @output : average : float
                       Average euclidean distance between connected points in the tree.
    @output : std     : float
                       Standard deviation of the euclidean distance between connected points in the tree.
    """
    V0, V1 = V.index_select(0,F[:,0]), V.index_select(0,F[:,1])
    u                                    =    (V1-V0)
    lengths                              =    (u**2).sum(1)[:, None].sqrt()


    LS = torch.zeros(V.shape[0], dtype=torchdtype, device=torchdeviceId)

    LS[F[:,0]]+= lengths[:,0]
    LS[F[:,1]]+=lengths[:,0]

    return (.5*LS).view(-1,1)


def Compute_structures(V,F):
    """
    For the linear normal kernel
    
    Parameters
    ----------
    @param : V : torch tensor
                 n-points x d-dimension points.
    @param : F : torch Long tensor
                 m-connections x 2-dim tensor containing pair of connected points' indices.

    Returns
    -------
    @output : centers        : torch tensor
                               npoints-1 x d-dimension points, the centers of each discretization segment in the tree.
    @output : lengths        : float
                               npoints-1 x 1-dimension tensor, the length of each discretization segment in the tree.
    @output : normalized_seg : torch tensor
                               npoints-1 x d-dimension normalized vectors of the discretization segments in tne tree. 
    """

    V0, V1 = V.index_select(0,F[:,0]), V.index_select(0,F[:,1])
    
    u                                    =    (V1-V0)
    lengths                              =    (u**2).sum(1)[:, None].sqrt()

    normalized_tgt_ok                       =     u / (lengths.view(-1,1))

    centers                               = (V0+V1)/2.
    return centers, lengths, normalized_tgt_ok



#####################################################################
########################### DATALOSSES ##############################
#####################################################################

#%% Dataloss for curves
accepted_methods = ['Varifold', 'PositionVarifold']
class CurvesDataloss():
    
    def __init__(self, method, source_connections, target, target_connections,
                 sigmaW, s_con_selected = None, t_con_selected = None, VS=None, target_curves_map=None):
        """
        Initialize the dataloss function to compare curves or graph of curves. 
        These parameters are shared by all datalosses.
        
        Parameters
        ----------
        @param : method             : string, the name of the attachment called. 
                                      Default : Varifold.
                                      Currently accepted methods : 
                                          Varifold, 
                                          PositionVarifold

                                          
        @param : source_connections : torch Long tensor 
                                      Pairs of integers : the source tree connected points' indices.
        @param : target             : torch tensor
                                      n-points x d-dimension vertices of the target.
        @param : target_connections : torch Long tensor 
                                      Pairs of integers : the target tree connected points' indices.
        @param : sigmaW             : torch float tensor, 
                                      positive scale of the data attachment term.

        @param  : s_con_selected    : torch Long tensor, 
                                      The selected connections in the source tree. Default : None.
                                      If s_con_selected is not None, the data attachment will be computed on a subtree of the source tree.
        @param  : t_con_selected    : torch Long tensor, 
                                      The selected connections in the target tree. Default : None.
                                      If FS_sel is not None, the data attachment will be computed on a subtree of the target tree.    
        """

        self.selected_source = True
        #If no selected connections provided, set selected to whole connections:
        if s_con_selected is None:  
            s_con_selected = source_connections
            self.selected_source = False
        if t_con_selected is None:  
            t_con_selected = target_connections   
    
        self.FS     = source_connections
        self.FS_sel = s_con_selected
        
        self.FT     = target_connections
        self.FT_sel = t_con_selected
        self.VT     = target
        
        self.sigmaW = sigmaW
        self.method = method
        self.VS     = VS    

        self.target_curves_map = target_curves_map

    
    def set_method(self,method):
        """
        Parameters
        ----------
        @param : method : string, the name of the attachment called. 
                            Default : 'Varifold'.
                            Currently accepted methods : 
                                Varifold, 
                                PositionVarifold, 

        """

        if method in accepted_methods:
            self.method = method
        else:
            print('The required method {0} is not accepted, please use one of : {1}'.format(method,accepted_methods))
            print('Using default : Varifold')
            self.method = 'Varifold'

        return
        
        
    def set_FS(self, FS):
        self.FS_sel = FS


    #Function to regroup the data attachment calling.
    def data_attachment(self):
        """
        Return the called dataloss function depending on the initialized method.                                   
                                              
        Returns
        -------                                      
        @output : dataloss  : function of the source tree points position. 
        """
        
        if self.method == "PositionVarifold":
            print("Using Position Varifold, for tree space optimization")
            dataloss = self.PositionVarifold()

        else:
            if(self.method!="Varifold"):
                print("Specified method not accepted, using default data attachment : Varifold")
            print("Using Varifold")
            dataloss = self.VarifoldCurve()
    
        return dataloss
    
    
    #%% Treespace losses
    def PositionVarifold(self):
        """
        The default dataloss : Varifold. 
        
        Returns
        -------
        @output : loss : the data attachment function.
        """
        K = OrientedGaussLinKernel(sigma=self.sigmaW) #
        
        CT, LT, NTn = Compute_structures(self.VT, self.FT_sel)
        cst = (LT * K(CT, CT, NTn, NTn, LT)).sum()

        def loss(V, F):
            self.FS = F
            self.FS_sel = F
            
            CS, LS, NSn = Compute_structures(V, F)     
            cost = 10*np.pi**2/4*(cst + (LS * K(CS, CS, NSn, NSn, LS)).sum() 
                   - 2 * (LS * K(CS, CT, NSn, NTn, LT)).sum())
            return cost/(self.sigmaW**2)
                        
        return loss
    
    
    #%% Varifold datalosses
    def VarifoldCurve(self):
        """
        The default dataloss : Varifold. 
        
        Returns
        -------
        @output : loss : the data attachment function.
        """
        K = OrientedGaussLinKernel(sigma=self.sigmaW)
        
        CT, LT, NTn = Compute_structures(self.VT, self.FT_sel)
        cst = (LT * K(CT, CT, NTn, NTn, LT)).sum()

        def loss(VS):
                        
            CS, LS, NSn = Compute_structures(VS, self.FS_sel)     
            cost = 10*np.pi**2/4*(cst + (LS * K(CS, CS, NSn, NSn, LS)).sum() 
                   - 2 * (LS * K(CS, CT, NSn, NTn, LT)).sum())
            return cost/(self.sigmaW**2)
        return loss
    
    
