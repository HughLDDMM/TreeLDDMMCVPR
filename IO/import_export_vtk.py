######################################SOME INPUT/OUTPUT FUNCTIONS (import/export data)#######
from pyvtk import PolyData, PointData, CellData, Scalars, VtkData, Vectors
import os.path
import sys
from time import gmtime, strftime
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)
import numpy as np
import json
#import torch
#from torch.autograd import Variable, grad, gradcheck       
 
###### Import the data ############################
def import_vtk(fname,*args, dim=3, **kwargs):
    data   = VtkData(fname)
    if np.shape(data.structure.polygons)[1] == 0:
        connec = np.array(data.structure.lines)
    else: 
        connec = np.array(data.structure.polygons)
    
    index  = connec.shape[1]
    if dim is None : dim = index  # By default, a curve is assumed to be 2D, surface 3D
    points = np.array(data.structure.points)[:,0:dim]
    
    try:
        labels = data.point_data.data[0].scalars
    except AttributeError:
        return points, connec, [-1]
    
    return points, connec, labels # torch.from_numpy( points ).type(torch.FloatTensor), torch.from_numpy( connec ), labels

def import_vectors(fname,*arg,dim=3,**kwargs):
    
    data   = VtkData(fname)
    if np.shape(data.structure.polygons)[1] == 0:
        connec = np.array(data.structure.lines)
    else: 
        connec = np.array(data.structure.polygons)
    
    index  = connec.shape[1]
    if dim is None : dim = index  # By default, a curve is assumed to be 2D, surface 3D
    points = np.array(data.structure.points)[:,0:dim]
    
    vectors = data.point_data.data[0].vectors

    return points, connec, vectors # torch.from_numpy( points ).type(torch.FloatTensor),


def import_momenta(fname,*arg,dim=3,**kwargs):
    
    data   = VtkData(fname) 
    points = np.array(data.structure.points)[:,0:dim] # By default 3D  
    vectors = data.point_data.data[0].vectors

    return points, vectors

#### export the data in vtk format#################   
    
def export_vtk(V,F, filename,path) :
    structure = PolyData(points  =      V.tolist(),
                         polygons = F.tolist())

    vtk = VtkData(structure)
    fname = filename +".vtk" ; os.makedirs(path, exist_ok=True)
    vtk.tofile( path+os.path.sep +fname,'ascii' )

    return

def export_labeled_vtk(V,F,L,filename,path):
    
    structure = PolyData(points  =      V.tolist(),
                         polygons = F.tolist())

    labels = PointData(Scalars(L,name='Labels'))

    vtk = VtkData(structure,labels)
    fname = filename +".vtk" ; os.makedirs(path, exist_ok=True)
    vtk.tofile( path+os.path.sep +fname,'ascii' )
    
    return

def export_multilabeled_vtk(V,F,L,filename,path):
    """
    Export to path/filename.vtk a file containig points, connections between points and the points labels
    @param : V        : list of lists of coordinates in R^d.
    @param : F        : list of 2-integer lists, the paired points.
    @param : L        : list of lists list of labels, the points labels and/or features. 
    @param : filename : string : nb, no need for .vtk at the end, will be automatically appended. 
    @param : path     : string, location to save. 
    """
    structure = PolyData(points  =      V.tolist(),
                         polygons = F.tolist())

    labels = PointData(Scalars(L[0],name='Labels'))

    vtk = VtkData(structure,labels)
    
    for i,lab in enumerate(L[1:]):
        vtk.append( PointData( Scalars( lab,name='Labels'+str(i+1) ) ) )
    
    fname = filename +".vtk" ; os.makedirs(path, exist_ok=True)
    vtk.tofile( path+os.path.sep +fname,'ascii' )
    
    return


def export_paired_labeled_vtk(V1,F1,L1,V2,F2,L2,pairing,filename,path):
    
    print(V1.shape)
    V1 = V1.tolist()+V2.tolist()
    print(len(V1))
    
    F1 = F1.tolist()+F2.tolist()
    F1+=pairing
    structure = PolyData(points  =  V1,
                         polygons = F1)

    labels = PointData(Scalars(list(L1)+list(L2),name='Labels'))

    vtk = VtkData(structure,labels)
    fname = filename +".vtk" ; os.makedirs(path, exist_ok=True)
    vtk.tofile( path+os.path.sep +fname,'ascii' )
    
    return

def export_pairs_vtk(points,pairs,L,filename,path):
    """
    Export to vtk points and pairs at path/filename +'.vtk' location.
    @param : points   : list of lists of coordinates in R^d.
    @param : pairs    : list of 2-integer lists, the paired points.
    @param : L        : the labels of the pairs (e.g the length).
    @param : filename : string : nb, no need for .vtk at the end, will be automatically appended. 
    @param : path     : string, location to save. 
    """
    structure = PolyData(points  =  points,
                         polygons = pairs)

    labels = PointData(Scalars(L,name='Labels'))

    vtk = VtkData(structure,labels)
    fname = filename +".vtk" ; os.makedirs(path, exist_ok=True)
    vtk.tofile( path+os.path.sep +fname,'ascii' )
    
    return


def export_vector_features(V,F, vectors, filename, path):
    
    structure = PolyData(points  =      V.tolist(),
                         polygons = F.tolist())
    v_field = PointData(Vectors(vectors,name='features'))

    vtk = VtkData(structure,v_field)
    fname = filename +".vtk" ; os.makedirs(path, exist_ok=True)
    vtk.tofile( path+os.path.sep +fname,'ascii' )
    
    return
    
def export_vector_field(V,F, vectors, filename, path):
    
    structure = PolyData(points  =      V.tolist(),
                         polygons = F.tolist())
    v_field = PointData(Vectors(vectors,name='momentums'))

    vtk = VtkData(structure,v_field)
    fname = filename +".vtk" ; os.makedirs(path, exist_ok=True)
    vtk.tofile( path+os.path.sep +fname,'ascii' )
    
    return


def export_momenta(V, vectors, filename, path):
    
    structure = PolyData(points = V.tolist())
    v_field = PointData(Vectors(vectors,name='momentums'))

    vtk = VtkData(structure,v_field)
    fname = filename +".vtk" ; os.makedirs(path, exist_ok=True)
    vtk.tofile( path+os.path.sep +fname,'ascii' )
    
    return
    
    
    
def export_points(V, filename, path):
    
    structure = PolyData(points = V.tolist())

    vtk = VtkData(structure)
    fname = filename +".vtk" ; os.makedirs(path, exist_ok=True)
    vtk.tofile( path+os.path.sep +fname,'ascii' )
    
    return
    
### Export the trajectory of the source + the target in vtk format
def export_result(x_traj,VT,FT,FS,q0,summary,path,filename):
    """
    Save the results to .vtk files. It saves the trajectory 
    """

    date =  strftime("%b_%d_%Y_%H_%M", gmtime())
    try:
        os.mkdir(path)
    except OSError:
        pass
    path = path + os.path.sep + date
    try:
        os.mkdir(path)
    except OSError:
        pass
    
    print(summary)

    f = open(path+os.path.sep+"summary.txt","w+")
    f.write(json.dumps(summary))
    f.close()
    n_traj = len(x_traj)
    nq,d = q0.shape
    for i in range(n_traj):
        print("STEP : ",i)
        xi = x_traj[i]
        export_vtk(xi,FS,"shoot_"+str(i),path + os.path.sep + filename)
    export_vtk(VT,FT,"target",path + os.path.sep + filename)      

  
def vtk2npz(directory, fname, dim=3, saving_directory = ""):
    
    if saving_directory == "":
        saving_directory = directory
        
    points, connections, labels = import_vtk(directory+'/'+fname, dim=dim)
    
    points = np.asarray(points)
    connections = np.asarray(connections)
    labels = np.asarray(labels)

    with open(saving_directory+'/'+fname+'.npz', 'wb') as outfile:
        np.savez(outfile, points=points, connections=connections, labels=labels)
    #outfile.close()
    
    return 
    
