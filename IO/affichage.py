#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:41:33 2019
"""

from IO import read_tree_depth
# =============================================================================
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from mpl_toolkits.mplot3d import Axes3D

import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg

# =============================================================================

import numpy as np
from definitions import artery_labels

overlap = {name for name in mcd.CSS4_COLORS
           if "xkcd:" + name in mcd.XKCD_COLORS}
overlap = sorted(overlap, reverse=True)

from import_export_vtk import export_pairs_vtk

def plot_PA_trees(tree_list,name_list,color_list = [], PathToSave="",show = True):
    """
    Plot the trees from the prostate database.

    tree_list : a list of list of points to plot.
    name_list : list containing the trees names. Mus be same length as 
                tree_list
    color_list : list containing the colors, if not of the same size, default 
                 cycling colors are used.
    
    If path_to_save != "", the plot is saved to the given path.
    """

    if(len(name_list)!=len(tree_list)):
        print("Input error : tree_list and name_list sizes must match")
        return
    
    if( (color_list==[]) or (len(color_list)!=len(tree_list)) ):
        print("color_list's length does not match tree_list's, set to default.")
        color_list = []
    save = False

    if(PathToSave!=""):
        save = True

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=200., azim=60)
       
    
    for cpt,points in enumerate(tree_list):
        if(color_list==[]):
            ax.scatter(points[:,0], points[:,1], points[:,2],s=2,
                       label=name_list[cpt])
        else:
            ax.scatter(points[:,0], points[:,1], points[:,2],c=color_list[cpt],
                       s=2,label=name_list[cpt])
    
    plt.legend(loc=2)     
    if(save):
        plt.savefig(PathToSave)

    if(show):
        plt.show()

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=200., azim=120)
       
    
    for cpt,points in enumerate(tree_list):
        if(color_list==[]):
            ax.scatter(points[:,0], points[:,1], points[:,2],s=2,
                       label=name_list[cpt])
        else:
            ax.scatter(points[:,0], points[:,1], points[:,2],c=color_list[cpt],
                       s=2,label=name_list[cpt])
    
    plt.legend(loc=2)     
    if(save):
        plt.savefig(PathToSave+"_other_view")

    if(show):
        plt.show()

    return


def plot_labeled_tree(tree,name,labels):
    """
    Plot a tree from the prostate database.

    tree   : list   : of points to plot.
    name   : string : the tree name. 
    labels : list   : containing the label of each point.
    
    """

    assert len(tree)==len(labels),"Input error : tree and labels sizes must match"

    
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=200., azim=60)
       
    for label in np.unique(labels):
        
        ax.scatter(tree[labels==label,0], 
                   tree[labels==label,1], 
                   tree[labels==label,2],s=2,label=artery_labels[label])

    
    plt.legend(loc=2)     
    plt.show()

    return





def make_deformation_GIF(target,trajectory,PathToSave):
    """
    source : the tree to deform
    target : target tree
    trajectory : list of intermediate deformed trees
    """

    nt = len(trajectory)

    max_x = max(trajectory[0][:,0].max(),target[:,0].max())
    max_y = max(trajectory[0][:,1].max(),target[:,1].max())
    max_z = max(trajectory[0][:,2].max(),target[:,2].max())
    
    min_x = min(trajectory[0][:,0].min(),target[:,0].min())
    min_y = min(trajectory[0][:,1].min(),target[:,1].min())
    min_z = min(trajectory[0][:,2].min(),target[:,2].min())

    images = []
    for t in range(nt):
        qnp = trajectory[t]
        #create fig        
        fig = plt.figure(figsize=(6,5), dpi=100)
        #link canvas to fig
        canvas = FigureCanvasAgg(fig)

        ax = Axes3D(fig)
        ax.view_init(elev=200., azim=60)

        ax.set_xlim3d(min_x, max_x)
        ax.set_ylim3d(min_y,max_y)
        ax.set_zlim3d(min_z,max_z)

        ax.w_xaxis.set_pane_color((1.0,1.0,1.0,1.0))
        ax.w_yaxis.set_pane_color((1.0,1.0,1.0,1.0))
        ax.w_zaxis.set_pane_color((1.0,1.0,1.0,1.0))

        ax.scatter(qnp[:,0], qnp[:,1], qnp[:,2],label='deformed',color='red',s=2)
        ax.scatter(target[:,0], target[:,1], target[:,2],label='target',color='green',s=2)
        
        ax.legend()
        ax.set_title('LDDMM matching example, step' + str(t))
        canvas.draw()

        #Save plot in a numpy array through buffer
        if(t==0):
            s, (width0,height0) = canvas.print_to_buffer()
        s, (width,height) = canvas.print_to_buffer()
        images.append(np.frombuffer(s, np.uint8).reshape((height0,width0,4)))


    imageio.mimsave(PathToSave+'.gif',images,duration=.5)
    
    plt.close('all')

    images = []
    for t in range(nt):
        qnp = trajectory[t]
        #create fig        
        fig = plt.figure(figsize=(6,5), dpi=100)
        #link canvas to fig
        canvas = FigureCanvasAgg(fig)

        ax = Axes3D(fig)
        ax.view_init(elev=200., azim=150)

        ax.set_xlim3d(min_x, max_x)
        ax.set_ylim3d(min_y,max_y)
        ax.set_zlim3d(min_z,max_z)

        ax.w_xaxis.set_pane_color((1.0,1.0,1.0,1.0))
        ax.w_yaxis.set_pane_color((1.0,1.0,1.0,1.0))
        ax.w_zaxis.set_pane_color((1.0,1.0,1.0,1.0))

        ax.scatter(qnp[:,0], qnp[:,1], qnp[:,2],label='deformed',color='red',s=2)
        ax.scatter(target[:,0], target[:,1], target[:,2],label='target',color='green',s=2)
        
        ax.legend()
        ax.set_title('LDDMM matching example, step' + str(t))
        canvas.draw()

        #Save plot in a numpy array through buffer
        if(t==0):
            s, (width0,height0) = canvas.print_to_buffer()
        s, (width,height) = canvas.print_to_buffer()
        images.append(np.frombuffer(s, np.uint8).reshape((height0,width0,4)))


    imageio.mimsave(PathToSave+"_view2.gif",images,duration=.5)

    return


n = 3
d = 3


def points_to_plot(points,max_plot):

    #tot = len(points)
    step = int(len(points)/max_plot)

    if(step==0):
        return points
    else:
        return points[0::step,:]


def save_pairings(tree1,tree2,step, savename, savepath):
    """
    Given two trees, every -step- points in tree1, pairs the closest point in
    tree2 and save the pairing vectors in a vtk file, with the length as label. 
    @param : tree1    : list of points
    @param : tree2    : list of points
    @param : savename : string, name of the .vtk file containing the pairings
    @param : savepath : string, path to the folder to save the pairing result
    """

    paired_points = []
    pairs = []
    dists = []

    def closest_point(point,tree):
        
        dist = np.Inf
        for p in tree:
            dist_temp = np.linalg.norm(np.asarray(point)-np.asarray(p))
            if(dist_temp < dist):
                closest_point = p
                dist = dist_temp
        
        return closest_point,dist

    for i in range(0,len(tree1),step):
        paired_points.append(tree1[i])

    n_points = len(paired_points)

    cpt = 0
    for i in range(0,len(tree1),step):
        pairs.append([cpt,n_points+cpt])
        associated_point,dist = closest_point(tree1[i],tree2)
        paired_points.append(associated_point)
        dists.append(dist)
        
        cpt+=1
        
    dists += [d for d in dists]
    
    export_pairs_vtk(paired_points,pairs,dists,savename,savepath)
    
    return 0


def save_label_pairings(tree1,l1,tree2,l2,step, savename, savepath):
    """
    Given two trees, every -step- points in tree1, pairs the closest point in
    tree2 and and save the pairing vectors in a vtk file. The pairs labels is 1
    if the paired points have the same anatomical label. It is 0 otherwise. 
    @param : tree1    : list of points
    @param : tree2    : list of points
    @param : savename : string, name of the .vtk file containing the pairings
    @param : savepath : string, path to the folder to save the pairing result
    """

    paired_points = []
    pairs = []
    lab_match = []

    def closest_point(point,tree,lab):
        
        sel_lab = 0
        dist = np.Inf
        for i,p in enumerate(tree):
            dist_temp = np.linalg.norm(np.asarray(point)-np.asarray(p))
            if(dist_temp < dist):
                closest_point = p
                dist = dist_temp
                sel_lab = lab[i]
        
        return closest_point,dist,sel_lab

    for i in range(0,len(tree1),step):
        paired_points.append(tree1[i])

    n_points = len(paired_points)

    cpt = 0
    for i in range(0,len(tree1),step):
        pairs.append([cpt,n_points+cpt])
        associated_point,dist,sel_lab = closest_point(tree1[i],tree2,l2)
        paired_points.append(associated_point)
        
        if(sel_lab == l1[i]):
            lab_match.append(1)
        else:
            lab_match.append(0)
        
        cpt+=1
        
    lab_match += [d for d in lab_match]
    
    export_pairs_vtk(paired_points,pairs,lab_match,savename,savepath)
    
    return 0

