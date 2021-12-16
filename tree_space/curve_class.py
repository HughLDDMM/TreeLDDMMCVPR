# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:21:25 2021
"""

from scipy.ndimage import zoom
import numpy as np
import copy

"""
methods : 
    get_length
    shorten
    interpolate2target
    
"""

class Curve(object):
    
    def __init__(self, instance = None, data = None, ind_first_point = 0):
        
        if instance is not None:
            #Create a copy of a Tree instance
            assert isinstance(instance,Curve), "The initial node must be a proper class Curve"
            self.__init__(data = copy.deepcopy(instance.data), 
                          ind_first_point = copy.deepcopy(instance.ind_first_point))  
            
        else:
            #Create a Tree instance from informations

            if data is not None:
                assert isinstance(data,(np.ndarray)), 'Right now, the only supported data are numpy ndarrays'      
                if len(data.shape)==1:
                    self.data = np.reshape(data, (1,-1))
                else:
                    self.data = data
            else:
                self.data = np.zeros((1,3))
        
            if self.data.shape[0]==1:
                self.points_connections = [[0]]
            else:
                self.points_connections = [[i,i+1] for i in range(self.data.shape[0]-1)] 

            self.ind_first_point = ind_first_point
            self.length = self.get_length()
            
            
    def __del__(self):
        del self
        
    
    def get_length(self, force = False):
        """
        Compute the length of the current branch.

        Parameters
        ----------
        force : Bool, optional
            Whether we want to recompute the length. Default is False.

        Returns
        -------
        length : float
            The length seen as the sum of the curve's segments' lengthes.

        """
        if self.data is None and self.length !=0 and force is False:
            print('No data, but still a given length, could be a Newick Tree. Returning self.length')
            return self.length
        
        n_pts = self.data.shape[0]
        if n_pts<=1 or len(self.data.shape)==1:
            return 0
        else:
            length = 0
            for i in range(n_pts-1):
                length += np.linalg.norm(self.data[i+1,:]-self.data[i,:])
                
            self.length = round(length, 3)
            return round(length, 3)
        
        
    def shorten(self, length2shorten, keep_size = False):
        """
        Shorten a current 

        Parameters
        ----------
        length2shorten : float
            DESCRIPTION.

        Returns
        -------
        None.

        """
                
        assert length2shorten <= self.get_length(), "The length to remove is greater that the branch length..."
        
        l = 0
        i = 1
        n,d = self.data.shape[0], self.data.shape[1]
        while l < length2shorten:
            l+= np.linalg.norm(self.data[n-i,:]-self.data[n-i-1,:])
            i+=1
            
        translation = self.data[-1,:]-self.data[n-i,:]
        self.data = (self.data[:n-i+1,:]).reshape(-1,d)

        if keep_size:
            self.resample_curve(n) 
        else:
            self.update_points_connections()
        
        self.length = self.get_length()

        return translation
        
    
    def update_points_connections(self):
        
        if self.data is not None:
            if self.data.shape[0]==1:
                self.points_connections = [[0]]
            else:
                self.points_connections = [[i,i+1] for i in range(self.data.shape[0]-1)] 
        return
    
    
    def interpolate2target(self, target_data, t):
        """
        Given a target branch, interpolate the current branch (seen as branch at t=0)
        and the target is the branch at t=1.
        
        Parameters
        ----------
        target_data : numpy ndarray
            The target branch.
            
        t           : float
            The time for the interpolation. 
            

        Returns
        -------
        None.
        
        
        """
        assert abs(self.data.shape[0]-target_data.shape[0])<=1, "self.data (shape : {0}) and target data (shape : {1}) must have 1 of difference in shape[0].".format(self.data.shape,target_data.shape)
        
        if t <= 1e-8:
            return
        
        if self.data.shape!=target_data.shape: 
            print("REDUCING THE BRANCH SIZE ... (source : {0}, target : {1})".format(self.data.shape[0],target_data.shape[0]))
            if self.data.shape[0]>target_data.shape[0]:
                self.data = self.data[1:,:]
            else:
                target_data = target_data[1:,:]
                        
        self.data = t*target_data + (1-t)*self.data
        
        return
        
    
    def resample_curve(self, n_points):
        """
        Resample a curve using zoom method. 
        
        Parameters
        ----------
        n_points : int
            The new number of points.
            
        """
        self.data[self.data[:,:]==0]=1e-10
        
        rescurve = zoom(self.data, (n_points/self.data.shape[0],1))

        rescurve[0,:]=self.data[0,:]
        rescurve[-1,:]=self.data[-1,:]
        
        self.data = rescurve

        self.update_points_connections()
        
        return 
    
    
    def update_length(self):
        """
        Compute the length of the current branch.

        Parameters
        ----------
        force : Bool, optional
            Whether we want to recompute the length. Default is False.

        Returns
        -------
        length : float
            The length seen as the sum of the curve's segments' lengthes.

        """

        n_pts = self.data.shape[0]

        if len(self.data.shape)==1:
            self.length = 0
        else:
            length = 0
            for i in range(n_pts-1):
                length += np.linalg.norm(self.data[i+1,:]-self.data[i,:])
                
            self.length = round(length, 3)

        return
