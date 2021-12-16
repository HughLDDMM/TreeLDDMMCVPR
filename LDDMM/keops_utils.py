#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 11:30:25 2019
"""
import os

import torch

from pykeops.torch import LazyTensor, Genred, Vi, Vj
from pykeops.torch.kernel_product.formula import *
from pykeops.torch import KernelSolve

################### GPU management #########################

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    l = []
    with open('tmp', 'r') as f:
        for x in f.readlines():
            l.append(int(x.split()[2]))
    
    memory_available = torch.tensor(l)
    f.close()
    return torch.argmax(memory_available)
    

# torch type and device
def TestCuda(verbose = True):
    use_cuda = torch.cuda.is_available()
    if(verbose):    
        print("Use cuda : ",use_cuda)
        
    if use_cuda:
        num_gpu = get_freer_gpu()
        torchdeviceId = torch.device('cuda:{0}'.format(num_gpu)) 
    else:
        torchdeviceId = 'cpu'
    torchdtype = torch.float64
    KernelMethod = 'CPU'
    
    if(use_cuda):
        torch.cuda.set_device(torchdeviceId)
        KernelMethod = 'auto'
    # PyKeOps counterpart
    KeOpsdeviceId = torchdeviceId.index  # id of Gpu device (in case Gpu is  used)
    KeOpsdtype = torchdtype.__str__().split('.')[1]  # 'float32'

    #print(KeOpsdtype)
    return use_cuda,torchdeviceId,torchdtype,KeOpsdeviceId,KeOpsdtype,KernelMethod

use_cuda,torchdeviceId,torchdtype,KeOpsdeviceId,KeOpsdtype,KernelMethod = TestCuda(verbose=False)


############################################################
# function to transfer data on Gpu only if we use the Gpu
def CpuOrGpu(x):
    if use_cuda:
        if type(x)==tuple:
            x = tuple(map(lambda x:x.cuda(device=torchdeviceId),x))
        else:
            x = x.cuda(device=torchdeviceId)
    return x


##################################################################
# Define the kernels
# ------------------
#
# Define Gaussian kernel :math:`K(x,y)_i = \sum_j \exp(-gamma*\|x_i-y_j\|^2)`


def GaussKernelWeight(sigma):

    """g = 1/(sigma**2)
    formula = 'b * Exp(-SqDist(x,y)*g)'
    variables = ['x = Vi(3)',
                 'y = Vj(3)',
                 'b = Vj(1)',
                 'g = Pm(1)'
                ]

    K = Genred(formula, variables, reduction_op='Sum', axis=1, dtype='float32')

    def f(x,y,b):
        return K(x,y,b,g)

    return f"""
    x, y, b = Vi(0,3), Vj(1,3), Vj(4,1)
    
    gamma = 1 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2*gamma).exp()
    
    return (K*b).sum_reduction(axis=1)


def GaussKernelWeight2D(sigma):

    g = 1/(sigma**2)
    formula = 'b * Exp(-SqDist(x,y)*g)'
    variables = ['x = Vi(2)',
                 'y = Vj(2)',
                 'b = Vj(1)',
                 'g = Pm(1)'
                ]

    K = Genred(formula, variables, reduction_op='Sum', axis=1, dtype='float32')

    def f(x,y,b):
        return K(x,y,b,g)

    return f
    

def GaussKernel(sigma):

    g = 1/(sigma**2)
    formula = 'b * Exp(-SqDist(x,y)*g)'
    variables = ['x = Vi(3)',
                 'y = Vj(3)',
                 'b = Vj(3)',
                 'g = Pm(1)'
                ]

    K = Genred(formula, variables, reduction_op='Sum', axis=1, dtype='float32')

    def f(x,y,b):
        return K(x,y,b,g)

    return f


def InvGaussKernel(sigma, ridge_coef = 0.01):
    
    g = 1/(sigma**2)
    formula = 'Exp(-SqDist(x,y)*g) * a * b'
    variables = ['x = Vi(3)',
                 'y = Vj(3)',
                 'a = Vj(1)',
                 'b = Vj(1)',
                 'g = Pm(1)'
                ]
                
    K = Genred(formula, variables, axis=1)
                
    def Kinv(x,b,c):
        
        _Kinv_ = KernelSolve(formula, variables, 'a', axis=1)
        res    = _Kinv_(x,x,c,b,g, alpha=ridge_coef)
        
        #print( "Error : ", ((ridge_coef*res + K(x,x,res,b,g) - c)**2).sqrt().sum() / len(x)    )
        
        return res
    
    return Kinv
    
    
def InvGaussKernel2D(sigma, ridge_coef = 0.01):
    
    g = 1/(sigma**2)
    formula = 'Exp(-SqDist(x,y)*g) * a * b'
    variables = ['x = Vi(2)',
                 'y = Vj(2)',
                 'a = Vj(1)',
                 'b = Vj(1)',
                 'g = Pm(1)'
                ]
                
    K = Genred(formula, variables, axis=1)
                
    def Kinv(x,b,c):
        
        _Kinv_ = KernelSolve(formula, variables, 'a', axis=1)
        res    = _Kinv_(x,x,c,b,g, alpha=ridge_coef)
        
        #print( "Error : ", ((ridge_coef*res + K(x,x,res,b,g) - c)**2).sqrt().sum() / len(x)    )
        
        return res
    
    return Kinv

###################################################################
# Define "Gaussian-CauchyBinet" kernel :math:`(K(x,y,u,v)b)_i = \sum_j \exp(-\|x_i-y_j\|^2) \langle u_i,v_j\rangle^2 b_j`

def GaussLinKernel(sigma):

    x, y, u, v, b = Vi(0,3), Vj(1,3), Vi(2,3), Vj(3,3), Vj(4,1)
    
    gamma = 1 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2*gamma).exp() * (u*v).sum()**2
    
    return (K*b).sum_reduction(axis=1)

    
def GaussLinKernel_matrix(sigma):

    x, y, u, v = Vi(0,3), Vj(1,3), Vi(2,3), Vj(3,3)
    
    gamma = 1 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2*gamma).exp() * (u*v).sum()**2
    
    return K


def OrientedGaussLinKernel(sigma):

    x, y, u, v, b = Vi(0,3), Vj(1,3), Vi(2,3), Vj(3,3), Vj(4,1)

    gamma = 1/(sigma**2)
    D2    = x.sqdist(y)
    K     = (-D2*gamma).exp() * ((u*v).sum()).exp()

    return (K*b).sum_reduction(axis=1)


def NormalizedGaussKernel(sigma,weights,nit=5):

    ker = GaussKernel(sigma)

    def K(x, y, b):
        s=torch.ones(weights.shape, dtype=torchdtype, device=torchdeviceId) 
        for i in range(nit):
            s = (1/(ker(x, y, weights)*s)).sqrt()
        return s*ker(params, x, y, s*b)
    return K


def SobolevKernel():

    D2 = x.sqdist(y)
    K = D2.pow(3/2)
    return (K).sum_reduction(axis=1)


def GenericGaussKernel(ref_scale,list_coefs = [1]):
    """

    Given a deformation scale for the kernels, generates the sum of the kernels at the scale divided by the coeff. 
    

    Parameters
    ----------
    ref_scale : torch tensor
        The scale of the deformations kernels (that is to be divided by the coefficients of list_coefs)
    list_coefs : list of floats
        The coefficients that will divide the deformations scale and ponder the kernels. 

    Returns
    -------
    labels1 : nparray (n_points)
        Updated labels of the points1

    """
    
    def K(x,y,b = None):
        """
        The kernel function. 

        Parameters
        ----------
        x,y : torch tensor
            The points x.shape = (n_pts_x, dim) y.shape = (n_pts_y, dim)
        b (Optional) : list of integers
            Optional momenta. Default: None 

        Returns
        -------
        a_i : LazyTensor

        If b is None, a_i = sum_j exp(-|x_i-y_j|^2)
        else,         a_i = sum_j exp(-|x_i-y_j|^2).b_i
 
        """

        x_i = LazyTensor( x[:,None,:] )  # x_i.shape = (n_pts_x, 1, dim)
        y_j = LazyTensor( y[None,:,:] )  # y_j.shape = ( 1, n_pts_y,dim)

        D_ij = ((x_i - y_j)**2).sum(dim=2)  # Symbolic (n_pts_x,n_pts_y,1) matrix of squared distances
        
        ref_scale2 = ref_scale**2

        weighting = 1/len(list_coefs)

        for i,coef in enumerate(list_coefs):

            c2 = coef**2
            
            if i==0:
                K_ij =  weighting*(- D_ij*c2/ref_scale2).exp()  # Symbolic (1e6,2e6,1)
            else:
                K_ij += weighting*(- D_ij*c2/ref_scale2).exp()  # Symbolic (1e6,2e6,1)

        if b is None:
            a_i = K_ij.sum(dim=1)  # Genuine torch.cuda.FloatTensor, a_i.shape = (1e6, 1),

        else:
            
            K_ij = K_ij * b[None,:,:]            
            a_i = K_ij.sum(dim=1)

        return a_i

    return K


def SumKernels(K_list):

    ponderation = 1/float(len(K_list))
    def K(x,y,b):

        for i,K_i in enumerate(K_list):
            if i==0:
                K_sum = torch.clone(K_i(x,y,b)).contiguous()
            else:
                K_sum+= K_i(x,y,b)
        return  ponderation*K_sum

    return K




###############################################################
################# SOME WRAPPING FUNCTIONS #####################
###############################################################

def Sum3GaussKernel(sigma = 100):

    list_coefs = [1., 2., 4.] 
    K = GenericGaussKernel(sigma,list_coefs)

    return K

def Sum4GaussKernel(sigma = 100):

    list_coefs = [1., 4., 8., 16.] 
    K = GenericGaussKernel(sigma,list_coefs)

    return K


def Sum4GaussKernel_bis(sigma = 100):

    list_coefs = [1., 2., 4., 8.] 
    K = GenericGaussKernel(sigma,list_coefs)

    return K

