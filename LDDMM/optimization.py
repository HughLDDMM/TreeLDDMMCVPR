######################### perform optimization ##############################
import torch
from scipy.optimize import minimize
import numpy as np
import time

import sys 
import os
sys.path.append(os.path.abspath("../IO"))
from import_export_vtk import export_momenta

from keops_utils import TestCuda

import pickle

params_opt=dict({"lr" : 1,"maxcor" : 10, "gtol" : 1e-3, "tol" : 1e-3, "use_scipy" : True, "method" : 'SLSQP'})
use_cuda,torchdeviceId,torchdtype,KeOpsdeviceId,KeOpsdtype,KernelMethod = TestCuda()


def opt(loss,p0,q0, maxiter = 100, folder2save = '',savename = ''):
    """
    Optimization function calling either scipy or torch method. 
    p0 is the variable to optimize, and can either be the initial momenta or a quaternion depending on the deformation one want to implement.
    """
    lr        = params_opt["lr"] 
    maxcor    = params_opt["maxcor"]
    gtol      = params_opt["gtol"]
    tol       = params_opt["tol"]
    use_scipy = params_opt["use_scipy"] #If use_scipy : perform otpimization with LBFGS on scipy. 
    method    = params_opt["method"]
    options   = dict( maxiter = maxiter,
                      ftol    = tol,
                      gtol    = gtol,
                      maxcor  = maxcor        # Number of previous gradients used to approximate the Hessian
                )

    loss_dict = {}

    loss_dict['A'] = [0]
    loss_dict['E'] = [0]

    optimizer = torch.optim.LBFGS([p0], line_search_fn='strong_wolfe')
    start = time.time()
    print('performing optimization...')
    opt.nit = -1
    def closure():
        opt.nit += 1; it = opt.nit

        optimizer.zero_grad()
        gamma,E,A = loss(p0,q0)

        L = gamma*E+A
        
        L.backward(retain_graph=True) #ATTENTION, CHANGE POUR ENCHAINER APRES RIGIDE, SINON ENLEVER RETAIN GRAPH !!! 

        print("Iteration ",it)

        if(folder2save != ''):
            if(opt.nit % 5 == 0):
                loss_dict['A'].append(float(A.detach().cpu().numpy()))
                loss_dict['E'].append(float(E.detach().cpu().numpy()))

        return L

    # Optimisation using scipy : we need to transfer the data from variable to float64
    def numpy_closure(vec):
        vec = lr*vec.astype('float64')
        numpy_to_model(p0,vec)
        c =  closure().data.view(-1).cpu().numpy()[0]
        dvec = model_to_numpy(p0,grad = True)
        return (c,dvec)
    
    def model_to_numpy(p, grad=False) :
        if grad :
            tensors = p.grad.data.view(-1).cpu().numpy()
        else :
            tensors = p.data.view(-1).cpu().numpy()     
        return np.ascontiguousarray( np.hstack(tensors) , dtype='float64' )
    
    def numpy_to_model(p, vec) :
        p.data = torch.from_numpy(vec).view(p.data.size()).type(p.data.type())

    if use_scipy :
        res = minimize( numpy_closure,      # function to minimize
                model_to_numpy(p0), # starting estimate
                method  = method,
                jac     = True,             # matching_problems also returns the gradient
                options = options)
        print(res.message)
        
    else :
        for i in range(int(maxiter/20)+1):            # Fixed number of iterations
            optimizer.step(closure)         # "Gradient descent" step.
    
    total_time = round(time.time()-start,2)
    print('Optimization time : ',total_time,' seconds')

    if(folder2save != ''):
        try:
            os.mkdir(folder2save)
        except OSError:
            pass

        loss_dict['Time'] = total_time
        loss_dict['it'] = opt.nit
        
        with open(folder2save+'/dict_'+savename+'.pkl','wb') as f:
            pickle.dump(loss_dict,f)
    return (p0,opt.nit,total_time)


def multiscale_opt(loss,p0,q0, maxiter = 100,folder2save = '',savename = ''):

    lr        = params_opt["lr"] 
    maxcor    = params_opt["maxcor"]
    gtol      = params_opt["gtol"]
    tol       = params_opt["tol"]
    use_scipy = params_opt["use_scipy"] #If use_scipy : perform otpimization with LBFGS on scipy. 
    method    = params_opt["method"]
    options   = dict( maxiter = maxiter,
                      ftol    = tol,
                      gtol    = gtol,
                      maxcor  = maxcor        # Number of previous gradients used to approximate the Hessian
                )

    loss_dict = {}

    loss_dict['A'] = []
    loss_dict['E'] = []
    loss_dict['E0'] = []
    loss_dict['E1'] = []
    loss_dict['E2'] = []
    loss_dict['E3'] = []

    optimizer = torch.optim.LBFGS([p0], line_search_fn='strong_wolfe')
    start = time.time()
    print('performing optimization...')
    opt.nit = -1
    def closure():
        opt.nit += 1; it = opt.nit

        optimizer.zero_grad()
        E_list,E,A = loss(p0,q0)

        L = E+A

        L.backward(retain_graph=True) #ATTENTION, CHANGE POUR ENCHAINER APRES RIGIDE, SINON ENLEVER RETAIN GRAPH !!! 

        print("Iteration ",it)
        print('E : ', E, " A : ", A)

        if(folder2save != ''):
            if(opt.nit % 5 == 0):
                loss_dict['A'].append(float(A.detach().cpu().numpy()))
                loss_dict['E'].append(float(E.detach().cpu().numpy()))
                for i,E_i in enumerate(E_list):
                    loss_dict['E'+str(i)].append(float(E_i.detach().cpu().numpy()))

        return L

    # Optimisation using scipy : we need to transfer the data from variable to float64
    def numpy_closure(vec):
        vec = lr*vec.astype('float64')
        numpy_to_model(p0,vec)
        c =  closure().data.view(-1).cpu().numpy()[0]
        dvec = model_to_numpy(p0,grad = True)
        
        return (c,dvec)
    
    def model_to_numpy(p, grad=False) :
        if grad :
            tensors = p.grad.data.view(-1).cpu().numpy()
        else :
            tensors = p.data.view(-1).cpu().numpy()     
        return np.ascontiguousarray( np.hstack(tensors) , dtype='float64' )
    
    def numpy_to_model(p, vec) :
        p.data = torch.from_numpy(vec).view(p.data.size()).type(p.data.type())
        #pdb.set_trace()

    #print(p0)

    if use_scipy :
        res = minimize( numpy_closure,      # function to minimize
                model_to_numpy(p0), # starting estimate
                method  = method,
                jac     = True,             # matching_problems also returns the gradient
                options = options)
        print(res.message)
        
    else :
        for i in range(int(maxiter/20)+1):            # Fixed number of iterations
            optimizer.step(closure)         # "Gradient descent" step.
    
    total_time = round(time.time()-start,2)
    print('Optimization time : ',total_time,' seconds')

    if(folder2save != ''):
        try:
            os.mkdir(folder2save)
        except OSError:
            pass
        
        with open(folder2save+'/dict_'+savename+'.pkl','wb') as f:
            pickle.dump(loss_dict,f)
    return (p0,opt.nit,total_time)


def template_opt(loss,P0,template, maxiter = 100):
    """
    Here P0 is the list of initial moments. 
    Template is also a variable.
    """

    lr        = params_opt["lr"] 
    maxcor    = params_opt["maxcor"]
    gtol      = params_opt["gtol"]
    tol       = params_opt["tol"]
    use_scipy = params_opt["use_scipy"] #If use_scipy : perform otpimization with LBFGS on scipy. 
    method    = params_opt["method"]
    options   = dict( maxiter = maxiter,
                      ftol    = tol,
                      gtol    = gtol,
                      maxcor  = maxcor        # Number of previous gradients used to approximate the Hessian
                )

    Variables = []
    for k,tensor in enumerate(P0):
        Variables+=[tensor]
    Variables+=[template]

    optimizer = torch.optim.LBFGS(Variables,max_eval=maxiter,lr=lr, line_search_fn='strong_wolfe')
    start = time.time()
    print('performing optimization...')
    opt.nit = -1
    def closure():
        opt.nit += 1; it = opt.nit
        optimizer.zero_grad()   

        L = loss(P0,template)

        L.backward(retain_graph=True) 
        print("Iteration ",it,", Cost = ", L.data.view(-1).cpu().numpy()[0])


        return L
    # Optpimisation using scipy : we need to transfer the data from variable to float64
    def numpy_closure(vec):      
        vec = lr*vec.astype('float64')
        numpy_to_model(Variables,vec)
        c =  closure().data.view(-1).cpu().numpy()[0]    
        dvec = model_to_numpy(Variables,grad = True)
        return (c,dvec)
    
    def model_to_numpy(Variables, grad=False) :
        if grad :
            tensors = [var.grad.data.view(-1).cpu().numpy() for var in Variables]          
            np.stack(tensors,axis=0)
        else :
            tensors = [var.data.view(-1).cpu().numpy() for var in Variables]          
            np.stack(tensors,axis=0)

        tensor = np.ascontiguousarray( np.hstack((tensors)) , dtype='float64' )        

        return tensor
    

    def numpy_to_model(torch_obj_list, np_obj) :
        """ Take the numpy 1d vector of parameters and reshape it into the different tensors (moment+template) """
        n_tensors = len(torch_obj_list)
        len_obj = np_obj.shape[0]/n_tensors
        assert len_obj==int(len_obj),'The numpy object size is no multiple of the number of tensors'
        len_obj=int(len_obj)

        for k,tensor in enumerate(torch_obj_list):
            tensor.data = torch.from_numpy(np_obj[k*len_obj:(k+1)*len_obj]).view(tensor.data.size()).type(tensor.data.type())
            
        #pdb.set_trace()

    #print(p0)

    if use_scipy :
        res = minimize( numpy_closure,      # function to minimize
                model_to_numpy(Variables), # starting estimate
                method  = method,
                jac     = True,             # matching_problems also returns the gradient
                options = options    )
        print(res.message)
        
    else :
        for i in range(int(maxiter/20)+1):            # Fixed number of iterations
            optimizer.step(closure)         # "Gradient descent" step.
    
    total_time = round(time.time()-start,2)
    print('Optimization time : ',total_time,' seconds')
    
    #if use_scipy:    
        #numpy_to_model(p0,res.x)

    #print(p0)

    return (Variables[:-1],Variables[-1],opt.nit,total_time)


def flow_opt(loss,x0,p0,q0, maxiter = 100,folder2save = '',savename = ''):

    lr        = params_opt["lr"] 
    maxcor    = params_opt["maxcor"]
    gtol      = params_opt["gtol"]
    tol       = params_opt["tol"]
    use_scipy = params_opt["use_scipy"] #If use_scipy : perform otpimization with LBFGS on scipy. 
    method    = params_opt["method"]
    options   = dict( maxiter = maxiter,
                      ftol    = tol,
                      gtol    = gtol,
                      maxcor  = maxcor        # Number of previous gradients used to approximate the Hessian
                )

    optimizer = torch.optim.LBFGS([p0], line_search_fn='strong_wolfe')
    start = time.time()
    print('performing optimization...')
    opt.nit = -1
    def closure():
        opt.nit += 1; it = opt.nit
        optimizer.zero_grad()
        L = loss(x0,p0,q0)
        L.backward(retain_graph=True) #ATTENTION, CHANGE POUR ENCHAINER APRES RIGIDE, SINON ENLEVER RETAIN GRAPH !!! 

        if(folder2save != ''):
            if(it==10 or it==50 or it==100 or it==500):
                temp = q0.detach().cpu().numpy()
                p0_np = p0.detach().cpu().numpy()
                export_momenta(temp, p0_np, 'Iter_'+str(it)+'_Momenta_'+savename, folder2save)

        return L

    # Optimisation using scipy : we need to transfer the data from variable to float64
    def numpy_closure(vec):
        vec = lr*vec.astype('float64')
        numpy_to_model(p0,vec)
        c =  closure().data.view(-1).cpu().numpy()[0]
        dvec = model_to_numpy(p0,grad = True)
        return (c,dvec)
    
    def model_to_numpy(p, grad=False) :
        if grad :
            tensors = p.grad.data.view(-1).cpu().numpy()
        else :
            tensors = p.data.view(-1).cpu().numpy()     
        return np.ascontiguousarray( np.hstack(tensors) , dtype='float64' )

    def numpy_to_model(p, vec) :
        p.data = torch.from_numpy(vec).view(p.data.size()).type(p.data.type())

    if use_scipy :
        res = minimize( numpy_closure,      # function to minimize
                model_to_numpy(p0), # starting estimate
                method  = method,
                jac     = True,             # matching_problems also returns the gradient
                options = options)
        print(res.message)
        
    else :
        for i in range(int(maxiter/20)+1):            # Fixed number of iterations
            optimizer.step(closure)         # "Gradient descent" step.
    
    total_time = round(time.time()-start,2)
    print('Optimization time : ',total_time,' seconds')
    
    return (p0,opt.nit,total_time)


def rigid_lddmm_opt(loss, quat0, p0, q0, maxiter = 100,folder2save = '',savename = ''):

    lr        = params_opt["lr"] 
    maxcor    = params_opt["maxcor"]
    gtol      = params_opt["gtol"]
    tol       = params_opt["tol"]
    use_scipy = params_opt["use_scipy"] #If use_scipy : perform otpimization with LBFGS on scipy. 
    method    = params_opt["method"]
    options   = dict( maxiter = maxiter,
                      ftol    = tol,
                      gtol    = gtol,
                      maxcor  = maxcor        # Number of previous gradients used to approximate the Hessian
                )

    optimizer = torch.optim.LBFGS([p0,quat0],max_eval=maxiter,lr=lr, line_search_fn='strong_wolfe')
    start = time.time()
    print('performing optimization...')
    opt.nit = -1

    loss_dict = {}

    loss_dict['A'] = [0]
    loss_dict['E'] = [0]
    loss_dict['E100'] = [0]
    loss_dict['E50'] = [0]
    loss_dict['E25'] = [0]
    loss_dict['E12'] = [0]

    def closure():
        opt.nit += 1; it = opt.nit
        optimizer.zero_grad()   

        (gamma,E100,E50,E25,E12,A,rotation_cost) = loss(quat0,p0,q0)

        E = E100+4.*E50+16.*E25+64.*E12
        L = gamma*E+A+0.0001*rotation_cost

        L.backward(retain_graph=True)  #
        print("Iteration ",it,", Cost = ", L.data.view(-1).cpu().numpy()[0])
        #print('Grad : ',quat0.grad)
        #print('QUAT0 : ', quat0)

        if(folder2save != ''):
            if(opt.nit % 5 == 0):
                loss_dict['A'].append(float(A.detach().cpu().numpy()))
                loss_dict['E'].append(float(E.detach().cpu().numpy()))
                loss_dict['E100'].append(float(E100.detach().cpu().numpy()))
                loss_dict['E50'].append(float(E50.detach().cpu().numpy()))
                loss_dict['E25'].append(float(E25.detach().cpu().numpy()))
                loss_dict['E12'].append(float(E12.detach().cpu().numpy()))

        return L
    # Optpimisation using scipy : we need to transfer the data from variable to float64
    def numpy_closure(vec):
        vec = lr*vec.astype('float64')
        numpy_to_model(quat0,vec[-7:])
        numpy_to_model(p0,vec[:-7].astype('float64'))
        c =  closure().data.view(-1).cpu().numpy()[0]     
        return (c,dvec)
    
    def model_to_numpy(p,quat, grad=False) :
        if grad :
            tensors = quat.grad.data.view(-1).cpu().numpy()
            p_tensors = p.grad.data.view(-1).cpu().numpy()
        else :
            tensors = quat.data.view(-1).cpu().numpy() 
            p_tensors = p.data.view(-1).cpu().numpy()   

        tensor = np.ascontiguousarray( np.hstack((p_tensors,tensors)) , dtype='float64' )        
        return tensor
    
    def numpy_to_model(torch_obj, np_obj) :
        torch_obj.data = torch.from_numpy(np_obj).view(torch_obj.data.size()).type(torch_obj.data.type())
        #pdb.set_trace()

    if use_scipy :
        res = minimize( numpy_closure,      # function to minimize
                model_to_numpy(p0,quat0), # starting estimate
                method  = method,
                jac     = True,             # matching_problems also returns the gradient
                options = options    )
        print(res.message)
        
    else :
        for i in range(int(maxiter/20)+1):            # Fixed number of iterations
            optimizer.step(closure)         # "Gradient descent" step.
    
    total_time = round(time.time()-start,2)
    print('Optimization time : ',total_time,' seconds')

    if(folder2save != ''):
        try:
            os.mkdir(folder2save)
        except OSError:
            pass
        
        with open(folder2save+'/dict_'+savename+'.pkl','wb') as f:
            pickle.dump(loss_dict,f)

    return (quat0,p0,opt.nit,total_time)


def rigid_opt(loss, quat0, q0, maxiter = 100,folder2save = '',savename = ''):

    lr        = params_opt["lr"] 
    maxcor    = params_opt["maxcor"]
    gtol      = params_opt["gtol"]
    tol       = params_opt["tol"]
    use_scipy = params_opt["use_scipy"] #If use_scipy : perform otpimization with LBFGS on scipy. 
    method    = params_opt["method"]
    options   = dict( maxiter = maxiter,
                      ftol    = tol,
                      gtol    = gtol,
                      maxcor  = maxcor        # Number of previous gradients used to approximate the Hessian
                )

    optimizer = torch.optim.LBFGS([quat0], max_eval=20, lr=lr, line_search_fn='strong_wolfe')
    start = time.time()
    print('performing optimization...')
    opt.nit = -1

    loss_dict = {}

    loss_dict['L'] = [0]

    def closure():
        opt.nit += 1; it = opt.nit
        optimizer.zero_grad()   

        L = loss(quat0, q0)

        L.backward(retain_graph=True)  #
        print("Iteration ",it,", Cost = ", L.data.view(-1).cpu().numpy()[0])

        if(folder2save != ''):
            if(opt.nit % 5 == 0):
                loss_dict['L'].append(float(L.detach().cpu().numpy()))

        return L
    # Optpimisation using scipy : we need to transfer the data from variable to float64
    def numpy_closure(vec):
        vec = lr*vec.astype('float64')
        numpy_to_model(quat0,vec)
        c =  closure().data.view(-1).cpu().numpy()[0]  
        dvec = model_to_numpy(quat0,grad = True)   
        return (c,dvec)
    
    def model_to_numpy(p, grad=False) :
        if grad :
            tensors = p.grad.data.view(-1).cpu().numpy()
        else :
            tensors = p.data.view(-1).cpu().numpy()     
        return np.ascontiguousarray( np.hstack(tensors) , dtype='float64' )
    
    def numpy_to_model(torch_obj, np_obj) :
        torch_obj.data = torch.from_numpy(np_obj).view(torch_obj.data.size()).type(torch_obj.data.type())
        #pdb.set_trace()

    if use_scipy :
        res = minimize( numpy_closure,      # function to minimize
                model_to_numpy(quat0), # starting estimate
                method  = method,
                jac     = True,             # matching_problems also returns the gradient
                options = options    )
        print(res.message)
        
    else :
        for i in range(int(maxiter)):            # Fixed number of iterations
            optimizer.step(closure)         # "Gradient descent" step.
    
    total_time = round(time.time()-start,2)
    print('Optimization time : ',total_time,' seconds')

    if(folder2save != ''):
        try:
            os.mkdir(folder2save)
        except OSError:
            pass
        
        with open(folder2save+'/dict_'+savename+'.pkl','wb') as f:
            pickle.dump(loss_dict,f)

    return (quat0,opt.nit,total_time)


