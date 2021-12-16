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

from linear_functions import *


####################################################################
# Custom ODE solver, for ODE systems which are defined on tuples
def RalstonIntegrator(nt=10):
    def f(ODESystem, x0, deltat=1.0):
        x = tuple(map(lambda x: x.clone(), x0))
        dt = deltat / nt
        for i in range(nt):
            xdot = ODESystem(*x)
            xi = tuple(map(lambda x, xdot: x + (2 * dt / 3) * xdot, x, xdot))
            xdoti = ODESystem(*xi)
            x = tuple(map(lambda x, xdot, xdoti: x + (.25 * dt) * (xdot + 3 * xdoti), x, xdot, xdoti))

        return x
    
    return f


#####################################################################
# Hamiltonian system

def Hamiltonian_points(K):
    def H(p, q):
        return .5 * (p * K(q, q, p))
    return H


def Hamiltonian(K):
    def H(p, q):
        return .5 * (p * K(q, q, p)).sum()
    return H


def HamiltonianSystem(K):
    H = Hamiltonian(K)
    def HS(p, q):
        Gp, Gq = grad(H(p, q), (p, q), create_graph=True)
        return -Gq, Gp
    return HS


def Shooting(p0, q0, K, deltat=1.0, Integrator=RalstonIntegrator(),FS=None):

    return Integrator(HamiltonianSystem(K), (p0, q0), deltat)


def Flow(x0, p0, q0, K, deltat=1.0, Integrator=RalstonIntegrator()):
    HS = HamiltonianSystem(K)
    def FlowEq(x, p, q):
        return (K(x, q, p),) + HS(p, q) #concatenation des fonctions pour le système hamiltonnien
    return Integrator(FlowEq, (x0, p0, q0), deltat)


def PositionLoss(dataloss):
    def loss(q0,connections):
        #dataloss.set_FS(connections)
        return dataloss(q0,connections)
    return loss


def LDDMMPositionLoss(K, dataloss, sigmaV, gamma=0.1):
    def loss(points, p0, q0, connections):
        #p,q = Shooting(p0, q0, K)
        x,p,q = Flow(points, p0, q0, K)
        Ev = Hamiltonian(K)(p0, q0)/(sigmaV**2)
        #dataloss.set_FS(connections)
        d = dataloss(x,connections)
        total = gamma*Ev + d
        print("Total cost : ", total)
        return total
    return loss


def LDDMMloss(K, dataloss,sigmaV, gamma=0.1):
    def loss(p0,q0):
        p,q = Shooting(p0, q0, K)
        Ev = Hamiltonian(K)(p0, q0)/(sigmaV**2)
        A = dataloss(q)
        print( 'Energy : {0}, Data attachment : {1}'.format(gamma*Ev.detach().cpu().numpy(),A.detach().cpu().numpy()))
        return gamma,Ev,A
        
    return loss



def Rigidloss(dataloss, gamma=0.1, rotate = False):
    """
    Loss function doing rigid deformations.
    """
    def loss(quat,q0):
        qrig = RigidDef(q0,quat,rotate=rotate)
        roll,pitch,yaw = Quaternion2EulerAngles(quat)
        rcost = RotationCost(roll,pitch,yaw)
        return dataloss(qrig) + rcost # QuaterNorm(quat) + 
    return loss



    

