# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:33:18 2021
"""

import copy
import numpy as np
import matplotlib.pyplot as plt

from VascularTree_class import plot_recursive
from phylogenetic_distance import get_leaves_contributions, get_common_edges, split_on_common, check_same_topology, get_splits
from RatioSequence import RatioSequence, interleave
from geodesic_tools import Geodesic, Geodesic_No_Common_Edges, follow_geodesic_interpolation


def getGeodesic(Tree1, Tree2):
    """
    Given a pair of trees, returns the geodesic.

    Parameters
    ----------
    T1        : VascularTree
        The first VascularTree from which we selected edges.
    T2        : VascularTree
        The second VascularTree from which we selected edges.
    edge_pair : list
        List of pairs of indices.

    Returns
    -------
    float
        The weight of a collection of edges.
    """
    
    #Pair the leaves between trees and compute their contribution to the geodesic
    leaves_contribution_squared = get_leaves_contributions(Tree1, Tree2)
        
    RS = RatioSequence()
    geod = Geodesic(RS)
    
    geod.set_leaf_contribution_squared(leaves_contribution_squared)
    
    #Find the common edges 
    common_edges_list, compatible1, compatible2, splits1, splits2 = get_common_edges(Tree1, Tree2)
    geod.set_common_edges(common_edges_list)
    
    list_no_common_pairs = split_on_common(Tree1, Tree2)
    
    #print(list_no_common_pairs)
    
    for i, pair_subtrees in enumerate(list_no_common_pairs):

        #print("\n \n Pair of subtrees : ",pair_subtrees)

        new_RS = Geodesic_No_Common_Edges(Tree1, Tree2, pair_subtrees[0], pair_subtrees[1])
        
        #combined_RS = interleave(geod.get_RS(), new_RS, Tree1, Tree2)
        #geod.set_ratio_sequence(combined_RS)
        geod.set_ratio_sequence(new_RS)

    return geod


def count_topologies(list_trees):
    """
    Count the number of unique topologies in a list of trees with the same set of leaves.
    
    """
    n_topo = 0
    uniques = []
    
    for i,t1 in enumerate(list_trees):

        if i==0:
            uniques.append(t1)
            n_topo += 1
        else:
            passed = True
            for t2 in uniques:
                if check_same_topology(t1, t2):
                    passed = False
                    break
            if passed:
                uniques.append(t1)
                n_topo += 1
    
    return n_topo, uniques


def count_splits(list_trees):
    """
    Count the possible splits in a list of trees, returns the unique splits and their associated nodes in the trees

    Parameters
    ----------
    list_trees            : list 
                            List of Vascular Trees from which we want to extract the uniques splits

    Returns
    -------
    splitting_nodes       : list 
                            List of nodes (Vascular Trees) corresponding to unique splits.  
    """

    n_topo, unique_topo = count_topologies(list_trees)

    list_splits     = []
    splitting_nodes = [] 

    leaves = set( (unique_topo[0]).get_leaves_labels() )

    for tree in unique_topo:
        splits_dict = get_splits(tree, intern = True)

        for node_id in splits_dict.keys():
            node_leaves   = set(splits_dict[node_id])
            #complementary = leaves - node_leaves

            if node_leaves not in list_splits: #if complementary not in list_splits and node_leaves not in list_splits:
                list_splits.append(node_leaves)
                splitting_nodes.append(tree.get_node(node_id))

    return splitting_nodes, list_splits


def InductiveMean(tree_list, plot = False):
    """
    This algorithm is not robust to the order of the trees in the tree_list !
    """
    
    barycenter = copy.deepcopy(tree_list[0])
    
    cpt = 0
    for target in tree_list[1:]:

        target_time = 1/(2+cpt)
        geod = getGeodesic(barycenter,target)
        barycenter = follow_geodesic_interpolation(geod, barycenter,target, target_time = target_time)
        cpt+=1
        
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title('Iteration {0}'.format(cpt))
            plot_recursive(barycenter,ax)
        
    return barycenter
    
    
def SturmMean(tree_list, plot = False, max_iter = 200, epsilon=1e-5, final_N = 5):
    """
    Provides an approximation of the mean tree with upper bound garantees.
    
    
    """
    
    np.random.seed(0)
    
    if final_N <= 1:
        print("final_N must be >=2, assigning 2")
        final_N = 2
    
    r = len(tree_list)
    
    init = np.random.randint(r)
    
    #print("IND INITIALIZATION :::::::::::::::::::::::::::::::::::: ", init)

    barycentree = copy.deepcopy(tree_list[init])

    cpt = 0
    displacement = 1e10
    
    vec_test = np.asarray([displacement for i in range(final_N)])
    
    while (cpt < max_iter and (vec_test > epsilon).any()):
    
        print("\n Iteration : ", cpt, "\n")
        ind_target = np.random.randint(r)
        target =  tree_list[ind_target]

        #print("IND TARGET :::::::::::::::::::::::::::::::::::: ", ind_target)

        target_time = 1/(2+cpt)
        geod = getGeodesic(barycentree,target)
        new_barycentree = follow_geodesic_interpolation(geod, barycentree, target, target_time = target_time)
        
        #print("NUMBER OF NODES : ", len(new_barycentree.get_nodes_list([])))

        displacement = geod.get_dist(barycentree,target)/(2+cpt)
        
        vec_test[:final_N-1] = vec_test[1:]
        vec_test[-1] = displacement
        
        cpt+=1
        barycentree = new_barycentree
        
        if cpt == max_iter:
            print("Maximum iterations reached, returning. \n")
        elif (vec_test <= epsilon).all():
            print("Tolerance threshold reached, returning. \n")

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('Iteration {0}'.format(cpt))
        plot_recursive(barycentree,ax)    
        
        
    return barycentree
    
