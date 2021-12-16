#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 11:30:25 2019
"""

# Standard imports

import torch
from torch.autograd import grad,Variable
import numpy as np
from keops_utils import *
torch.pi = torch.acos(torch.zeros(1)).item() * 2

import vedo
import vtk

####################################################################
# Rigid implementation
# --------------------

# Barycenter registration
def RawRegistration(VS,VT, use_torch = True):
    """
    A simple registration of the barycenters.
    Translate the source. 

    @param : VS : (n_source x dim) torch tensor, the source vertices (that are translated in the procedure)
    @param : VT : (n_target x dim) torch tensor, the target vertices

    @output : decalage : (1 x dim) torch tensor, the translation applied along each axis.  
    """

    if not use_torch :
        bary_S = np.mean(VS,0)
        bary_T = np.mean(VT,0)
    else:
        bary_S = torch.mean(VS,0)
        bary_T = torch.mean(VT,0)
    decalage = bary_T - bary_S

    for k in range(VS.shape[0]):
        VS[k,:]+=decalage

    return decalage


# Barycenter registration
def Rescale(VS, factor = 1.2, use_torch = True):
    """
    A simple registration of the barycenters.
    Translate the source. 

    @param : VS : (n_source x dim) torch tensor, the source vertices (that are translated in the procedure)
    @param : VT : (n_target x dim) torch tensor, the target vertices

    @output : decalage : (1 x dim) torch tensor, the translation applied along each axis.  
    """

    if not use_torch :
        bary_S = np.mean(VS,0)
    else:
        bary_S = torch.mean(VS,0)

    for k in range(VS.shape[0]):
        VS[k,:]-=bary_S

    VS*=factor
    
    for k in range(VS.shape[0]):
        VS[k,:]+=bary_S

    return VS


def RigidDefMatrix(source,deformation_matrix):
    """
    A standard rigid deformation stored in a 4x4 matrix,
    the scale part is not used yet.

    
    source : points,
    deformation_matrix : 4x4 matrix 
                        - [:3,:3] rotation
                        - [:3,3]  translation
                        - [3,3]   scale
    """

    xrig = torch.clone(source).t().contiguous()
    meanxrig = xrig.mean(1)
    xrig -= meanxrig.unsqueeze(1).expand_as(xrig)

    translat = deformation_matrix[:3,3]
    print('Translat : ', translat)
    Rq       = deformation_matrix[:3,:3]

    phix = Rq @ xrig

    phix += meanxrig.unsqueeze(1).expand_as(xrig)
    phix = torch.t(phix).contiguous()
    phix += translat 

    return phix


################### USING ICP FROM VTK #####################
def register_ICP(sourcePoints, targetPoints, maxIter = 100):
    """

    """

    source = vedo.Points(sourcePoints)
    target = vedo.Points(targetPoints)

    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source.polydata())
    icp.SetTarget(target.polydata())
    icp.GetLandmarkTransform().SetModeToRigidBody()

    icp.SetMaximumNumberOfIterations(maxIter)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()

    icpTransformFilter = vtk.vtkTransformPolyDataFilter()
    icpTransformFilter.SetInputData(source.polydata())

    icpTransformFilter.SetTransform(icp)
    icpTransformFilter.Update()

    transformedSource = icpTransformFilter.GetOutput()

    outputMatrix = vtk.vtkMatrix4x4()
    icp.GetMatrix(outputMatrix)
    def_rigid = vtkmatrix_to_numpy(outputMatrix)

    return vedo.Points(transformedSource), def_rigid

    
def vtkmatrix_to_numpy(matrix):
    """
    Copies the elements of a vtkMatrix4x4 into a numpy array.

    :param matrix: The matrix to be copied into an array.
    :type matrix: vtk.vtkMatrix4x4
    :rtype: numpy.ndarray
    """
    print('Some strange offset in the translation... To investiguate !')
    m = np.ones((4, 4))
    for i in range(4):
        for j in range(4):
            m[i, j] = matrix.GetElement(i, j)
    return m


################### QUATERNION BASED #####################
def RigidDef(source,quat,rescale=False, rotate=False):
    """
    Function Performing the scaling, translation and rotation from a quaternion of size 7 : 
    The coefficient are normalized anyway to create rigid transformations. 
    
    @param : source : (n_source x dim) torch tensor, the source vertices.
    @param : quat   : (7) torch tensor, the quaternion coefficients. 
                      quat[0]   : scale
                      quat[1:4] : rotations
                      quat[4:]  : translation

    @output : phix : torch tensor, the new points coordinates.
    """

    if rescale==False:
        scale_coef = torch.tensor(1, dtype=torchdtype, device=torchdeviceId, requires_grad=True) # quat[0]    

    xrig = torch.clone(source).t().contiguous()
    meanxrig = xrig.mean(1)
    xrig -= meanxrig.unsqueeze(1).expand_as(xrig)

    norm_rot_square = torch.pow(scale_coef, 2)+torch.pow(quat[1], 2)+torch.pow(quat[2], 2)+torch.pow(quat[3], 2)

    norm_rot = torch.sqrt(norm_rot_square)
    
    if rotate:
        m11 = torch.pow(scale_coef/norm_rot, 2)+torch.pow(quat[1]/norm_rot, 2)-torch.pow(quat[2]/norm_rot, 2)-torch.pow(quat[3]/norm_rot, 2)
        m12 = 2*(quat[1]/norm_rot*quat[2]/norm_rot - scale_coef/norm_rot*quat[3]/norm_rot)
        m13 = 2*(quat[1]/norm_rot*quat[3]/norm_rot + scale_coef/norm_rot*quat[2]/norm_rot)

        m21 = 2*(quat[1]/norm_rot*quat[2]/norm_rot + scale_coef/norm_rot*quat[3]/norm_rot)
        m22 = torch.pow(scale_coef/norm_rot, 2)+torch.pow(quat[2]/norm_rot, 2)-torch.pow(quat[1]/norm_rot, 2)-torch.pow(quat[3]/norm_rot, 2)
        m23 = 2*(quat[2]/norm_rot*quat[3]/norm_rot - scale_coef/norm_rot*quat[1]/norm_rot)

        m31 = 2*(quat[1]/norm_rot*quat[3]/norm_rot - scale_coef/norm_rot*quat[2]/norm_rot)
        m32 = 2*(quat[2]/norm_rot*quat[3]/norm_rot + scale_coef/norm_rot*quat[1]/norm_rot)
        m33 = torch.pow(scale_coef/norm_rot, 2)+torch.pow(quat[3]/norm_rot, 2)-torch.pow(quat[1]/norm_rot, 2)-torch.pow(quat[2]/norm_rot, 2)
        
        t = torch.stack((m11, m12, m13, m21, m22, m23, m31, m32, m33)) #/norm_rot_square

    if(use_cuda):
        if rotate:
            Rq = torch.reshape(t, (3,3)).cuda(device = torchdeviceId).contiguous()
        translat = torch.stack((quat[4],quat[5],quat[6])).cuda(device = torchdeviceId)
    else:
        if rotate:
            Rq = torch.reshape(t, (3,3)).contiguous()
        translat = torch.stack((quat[4],quat[5],quat[6]))

    if rotate:
        phix = Rq @ xrig
    else:
        phix = xrig
    phix += meanxrig.unsqueeze(1).expand_as(xrig)
    phix = torch.t(phix).contiguous()
    phix += translat 

    return phix


def Quat2Matrix(quat):

    scale_coef = torch.tensor(1, dtype=torchdtype, device=torchdeviceId, requires_grad=True)

    norm_rot_square = torch.pow(scale_coef, 2)+torch.pow(quat[1], 2)+torch.pow(quat[2], 2)+torch.pow(quat[3], 2)

    norm_rot = torch.sqrt(norm_rot_square)

    matrix = torch.eye(4)
    m11 = torch.pow(scale_coef/norm_rot, 2)+torch.pow(quat[1]/norm_rot, 2)-torch.pow(quat[2]/norm_rot, 2)-torch.pow(quat[3]/norm_rot, 2)
    m12 = 2*(quat[1]/norm_rot*quat[2]/norm_rot - scale_coef/norm_rot*quat[3]/norm_rot)
    m13 = 2*(quat[1]/norm_rot*quat[3]/norm_rot + scale_coef/norm_rot*quat[2]/norm_rot)

    m21 = 2*(quat[1]/norm_rot*quat[2]/norm_rot + scale_coef/norm_rot*quat[3]/norm_rot)
    m22 = torch.pow(scale_coef/norm_rot, 2)+torch.pow(quat[2]/norm_rot, 2)-torch.pow(quat[1]/norm_rot, 2)-torch.pow(quat[3]/norm_rot, 2)
    m23 = 2*(quat[2]/norm_rot*quat[3]/norm_rot - scale_coef/norm_rot*quat[1]/norm_rot)

    m31 = 2*(quat[1]/norm_rot*quat[3]/norm_rot - scale_coef/norm_rot*quat[2]/norm_rot)
    m32 = 2*(quat[2]/norm_rot*quat[3]/norm_rot + scale_coef/norm_rot*quat[1]/norm_rot)
    m33 = torch.pow(scale_coef/norm_rot, 2)+torch.pow(quat[3]/norm_rot, 2)-torch.pow(quat[1]/norm_rot, 2)-torch.pow(quat[2]/norm_rot, 2)

    t = torch.stack((m11, m12, m13, m21, m22, m23, m31, m32, m33))

    if(use_cuda):
        Rq = torch.reshape(t, (3,3)).cuda(device = torchdeviceId).contiguous()
        translat = torch.stack((quat[4],quat[5],quat[6])).cuda(device = torchdeviceId)
    else:
        Rq = torch.reshape(t, (3,3)).contiguous()
        translat = torch.stack((quat[4],quat[5],quat[6]))

    matrix[:3,:3] = Rq
    matrix[:3,3]  = translat

    return matrix

def RigidDefMatrix(source,deformation_matrix):
    """
    source : points,
    deformation_matrix : 4x4 matrix 
                        - [:3,:3] rotation
                        - [:3,3]  translation
                        - [3,3]   scale
    """


    xrig = torch.clone(source).t().contiguous()
    meanxrig = xrig.mean(1)
    xrig -= meanxrig.unsqueeze(1).expand_as(xrig)

    translat = deformation_matrix[:3,3]
    Rq       = deformation_matrix[:3,:3]

    if(use_cuda):
        translat.cuda(device = torchdeviceId)

    phix = Rq @ xrig

    phix += meanxrig.unsqueeze(1).expand_as(xrig)
    phix = torch.t(phix).contiguous()
    phix += translat 

    return phix


def Quaternion2EulerAngles(quat):
    """
    Given a unit quaternion, returns the rotations along (x,y,z) axis.
    Angles in radian !
    """

    roll_num   = 2*(quat[0]*quat[1]+quat[2]*quat[3])
    roll_denom = quat[0]*quat[0] - quat[1]*quat[1] - quat[2]*quat[2] + quat[3]*quat[3] 
    roll       = torch.atan2(roll_num,roll_denom)


    val = 2*(quat[0]*quat[2]-quat[3]*quat[1]) 
    if abs(val)>=1:
        pitch = torch.tensor(0) #copysign(np.pi/2,val) #90 degree if out of range
    else:
        pitch = torch.asin(val)

    yaw_num   = 2*(quat[0]*quat[3]+quat[1]*quat[2])
    yaw_denom = quat[0]*quat[0] + quat[1]*quat[1] - quat[2]*quat[2] - quat[3]*quat[3] # 1-2*(quat[2]*quat[2]+quat[3]*quat[3])

    #yaw_num   = 2*(quat[0]*quat[1]-quat[2]*quat[3])
    #yaw_denom = 1-2*(quat[1]*quat[1]+quat[3]*quat[3])

    yaw       = torch.atan2(yaw_num,yaw_denom)

    return roll,pitch,yaw


def RotationCost(roll,pitch,yaw):
    """
    The realistic rotations are in [-20°,20°] along each axis.
    (Corresponds to [-0.35,0.35] rad.

    Compute the rotation cost given such ranges.
    """

    b = torch.exp(torch.tensor(-7.))
    
    def sigmoid(x):

        s = 1/(1+b*torch.exp(-20.*torch.abs(x)))

        return s

    cost = (torch.tan(torch.pi/2.*sigmoid(torch.abs(roll)))+torch.tan(torch.pi/2.*sigmoid(torch.abs(pitch)))+torch.tan(torch.pi/2.*sigmoid(torch.abs(yaw)))).abs()

    return torch.tensor(cost, dtype=torchdtype, device=torchdeviceId, requires_grad=True)
    

class RigidMatch(torch.autograd.Function):
    """
    Own function for autograd. Forward the deformatioins from quaternion. 
    Backward : The hand-made gradient from the quaternion.
    """
    @staticmethod
    def forward(self,q0,quat):

        self.save_for_backward(q0,quat)
        output = RigidDef(q0,quat)
        return output

    @staticmethod
    def backward(self,grad_output):
        """
        Gradient loss wrt the output, need to compute gradient of loss wrt the input.

        Return : None : gradient relative to the points is not of interest.
                G : gradient relative to the quaternion.

        """
        

        q0,quat = self.saved_tensors

        xrig = torch.t(q0).contiguous()
        meanxrig = xrig.mean(1)
        xrig -= meanxrig.unsqueeze(1).expand_as(xrig)

        a = quat[1]

        ze = torch.tensor(0,dtype=torchdtype, device=torchdeviceId)

        temp = (ze,-quat[3],quat[2],quat[3],ze,-quat[1],-quat[2],quat[1],ze)

        ncross = torch.stack(temp)
        ncross = torch.reshape(ncross,(3,3))
        
        ndot = torch.stack((quat[1],quat[2],quat[3],quat[1],quat[2],quat[3],quat[1],quat[2],quat[3]))        
        ndot = torch.reshape(ndot,(3,3))

        Gp = torch.t(grad_output).contiguous()
        
        #gradient wrt rigid motion params
        G = torch.zeros(quat.shape, dtype=torchdtype, device=torchdeviceId)
        ncrossxrig = ncross @ xrig

        temp = (a * xrig + ncrossxrig)
        
        G[0] = 2*torch.sum(torch.sum(torch.mul(temp, Gp)))
        G[1:4] = 2*torch.sum(torch.cross(ncrossxrig,Gp) + a*torch.cross(xrig,Gp) + (ndot @ xrig)*Gp,1)
        G[4:] = torch.sum(Gp,1)

        G = G.data

        return None, G


def KernelMatrix(x,y,h):
    return h(np.sqrt(np.sum((x[:,None,:]-y[None,:,:])**2,axis=2)))

def RigidTPSMatching(S,T,K,scale=100,lbd=0):
    """

    """

    def TPSfun(r):
        return r**3/(16*np.pi)

    def GaussFun(r):
        return np.exp(-r*r/(scale)**2)+np.exp(-r*r/(scale/4)**2)+np.exp(-r*r/(scale/8)**2)+np.exp(-r*r/(scale/16)**2)
        

    h = GaussFun #TPSfun # 
    n,d = S.shape
    #Kss = K(S,S).detach().cpu().numpy()
    #print(Kss.shape)

    S_np = S.detach().cpu().numpy()
    T_np = T.detach().cpu().numpy()

    Kss = KernelMatrix(S_np,S_np,h)+lbd*np.eye(n)

    Z = np.zeros((n,n)) 

    L1_r = np.zeros((n,int(d*(d+1)/2)))
    L1_r[:,0] = 1
    L1_r[:,3] = S_np[:,1]
    L1_r[:,4] = S_np[:,2]
    L1 = np.concatenate((Kss,Z,Z,L1_r),axis=1)  
    
    L2_r = np.zeros((n,int(d*(d+1)/2)))
    L2_r[:,1] = 1 
    L2_r[:,3] = -S_np[:,0]
    L2_r[:,5] =  S_np[:,2]
    L2 = np.concatenate((Z,Kss,Z,L2_r),axis=1)

    L3_r = np.zeros((n,int(d*(d+1)/2)))
    L3_r[:,2] = 1  
    L3_r[:,4] = -S_np[:,0]
    L3_r[:,5] = -S_np[:,1]
    L3 = np.concatenate((Z,Z,Kss,L3_r),axis=1)

    Lr = np.concatenate((L1_r,L2_r,L3_r))
    print(Lr.shape)


    Lrt = np.concatenate(( Lr.T,np.zeros((int(d*(d+1)/2),int(d*(d+1)/2))) ),axis=1)

    M = np.concatenate((L1,L2,L3,Lrt))

    print(M.shape)

    c = T_np-S_np
    
    ct = np.zeros(int(d*n+d*(d+1)/2))
    ct[:n] = c[:,0]
    ct[n:2*n] = c[:,1] 
    ct[2*n:3*n] = c[:,2]   

    a = np.linalg.solve(M,ct)

    print("A affine : ", a[3*n:])

    def phi(x):
        #Kxs = K(x,S_np).detach().cpu().numpy()

        x_np = x.detach().cpu().numpy()
        nx = x_np.shape[0]

        Kxs = KernelMatrix(x_np,S_np,h)
        nx = x.shape[0] #x and S must have the same dimension...

        Zx = np.zeros(Kxs.shape)

        XL1_r = np.zeros((nx,int(d*(d+1)/2)))
        XL1_r[:,0] = 1
        XL1_r[:,3] = x_np[:,1]
        XL1_r[:,4] = x_np[:,2]
        XL1 = np.concatenate((Kxs,Zx,Zx,XL1_r),axis=1)  
        
        XL2_r = np.zeros((nx,int(d*(d+1)/2)))
        XL2_r[:,1] = 1 
        XL2_r[:,3] = -x_np[:,0]
        XL2_r[:,5] =  x_np[:,2]
        XL2 = np.concatenate((Zx,Kxs,Zx,XL2_r),axis=1)

        XL3_r = np.zeros((nx,int(d*(d+1)/2)))
        XL3_r[:,2] = 1  
        XL3_r[:,4] = -x_np[:,0]
        XL3_r[:,5] = -x_np[:,1]
        XL3 = np.concatenate((Zx,Zx,Kxs,XL3_r),axis=1)

        N = np.concatenate((XL1,XL2,XL3))

        dx = np.dot(N,a)
        
        delta_x = np.zeros(x.shape)
        delta_x[:,0] = dx[:nx]
        delta_x[:,1] = dx[nx:2*nx]
        delta_x[:,2] = dx[2*nx:]
        delta_x_torch = torch.tensor(delta_x, dtype=torchdtype, device=torchdeviceId)

        dx_aff = np.dot(N[:,d*n:],a[d*n:])
        delta_x_aff = np.zeros(x.shape)
        delta_x_aff[:,0] = dx_aff[:nx]
        delta_x_aff[:,1] = dx_aff[nx:2*nx]
        delta_x_aff[:,2] = dx_aff[2*nx:]
        delta_x_aff_torch = torch.tensor(delta_x_aff, dtype=torchdtype, device=torchdeviceId)

        return x+delta_x_torch, x+delta_x_aff_torch

    return phi

def MatchingTPS(S,T,K,scale=100,lbd=0):
    """

    """

    def TPSfun(r):
        return r**3/(16*np.pi)

    def GaussFun(r):
        return np.exp(-r*r/(scale)**2)+np.exp(-r*r/(scale/4)**2)+np.exp(-r*r/(scale/8)**2)+np.exp(-r*r/(scale/16)**2)
        

    h = GaussFun #TPSfun # 
    n,d = S.shape

    S_np = S.detach().cpu().numpy()
    T_np = T.detach().cpu().numpy()

    Kss = KernelMatrix(S_np,S_np,h)+lbd*np.eye(n)

    St = np.concatenate((np.ones((n,1)),S_np),axis=1)
    M1 = np.concatenate((Kss,St),axis=1)
    M2 = np.concatenate((St.T,torch.zeros((d+1,d+1))),axis=1)
    M = np.concatenate((M1,M2))

    c = T_np-S_np
    
    ct = np.concatenate((c,torch.zeros((d+1,c.shape[1]))))

    a = np.linalg.solve(M,ct)
    print(a[n:,:])

    def phi(x):
        x_np = x.detach().cpu().numpy()
        nx = x_np.shape[0]

        Kxs =  KernelMatrix(x_np,S_np,h) # K(x,S)

        print("applied : ", Kxs.shape)

        nx = x.shape[0] #x and S must have the same dimension...
        xt = np.concatenate((np.ones((nx,1)),x_np),axis=1)
        N  = np.concatenate((Kxs,xt),axis=1)

        return x+torch.tensor(np.dot(N,a), dtype=torchdtype, device=torchdeviceId), x+torch.tensor(np.dot(N[:,n:],a[n:,:]), dtype=torchdtype, device=torchdeviceId)

    return phi


def QuaterNorm(q):
    """
    Here we define the norm relative to the identity quaternion :
    (1,0,0,0,0,0,0)
    """
    
    temp = torch.tensor(q)

    scale_norm = (temp[0])**2
    scale_norm = -torch.log(scale_norm+0.00001)
    
    temp[0]+=scale_norm

    rot_norm = torch.sqrt((temp[1])**2+(temp[2])**2+(temp[3])**2)
    if(rot_norm!=0):
        temp[1]/=rot_norm
        temp[2]/=rot_norm
        temp[3]/=rot_norm

    return torch.norm(temp)




    

