# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 10:27:17 2021
"""

from VascularTree_class     import VascularTree, print_recursive
from main_geodesic         import getGeodesic, SturmMean, count_topologies, count_splits
from phylogenetic_distance import get_splits
import copy
import numpy as np


import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(100000)

import math 



# THE TREE CLASS
class SubBHV(VascularTree):
    """
    
    """
    
    
    def __init__(self, list_trees, active_tree = -1):
        """
        SubBHV
        
        From a set of Trees (vascular trees), build a set of orthants and possible topologies,
        with one active tree. 
        
        \mathcal{C} : 
        
        Parameters
        ----------
        list_trees            : list of VascularTrees
            Need at least one VascularTree to derive one orthant.
            
        active_tree           : int, optional
            The active tree in the list_trees. Corresponds to the position in the curent subspace of the BHV space.
            Default is -1. 
            
            If active_tree is -1, start with the Sturm Mean
            else active_tree must be >= 0, and < len(list_trees)


        Returns
        -------
        None (the class instance.)
        """
        
        assert len(list_trees)>0, 'Need at least one tree'
        assert all([isinstance(v, VascularTree) for v in list_trees]), 'All trees must be VascularTrees instances.'

        #TODO : Add a check on the set of leaves for each tree : must be the same...

        if active_tree == -1:
            self.position = SturmMean(list_trees, max_iter=200)
            self.position.remove_common_points()
        elif active_tree >= 0 and active_tree < len(list_trees):
            self.position = list_trees[active_tree]
        else:
            print("Invalid value for active_tree ({0}, must be between {1} and {2}.".format(active_tree, -1, len(list_trees)))
            return
            
        self.n_topo, self.unique_topos     = count_topologies(list_trees)
        self.leaves     = set(self.position.get_leaves_labels()) 

        tmp        = []

        for t in list_trees:
            t.resample_tree(2)
            tmp.append(t)

        self.tree_set = tmp

        """
        Number of points to create = 2*n_leaves-3 (current topology) + number of other splits in the other topologies
        These will be the extremities of the segments of the current tree + points added at the position in the current 
        tree of points containing all the offspring leaves of the point to add. 
        """

        #We save the splits for latter operations such as adding or removing a tree from tree set
        self.splitting_nodes, self.list_splits = count_splits(self.unique_topos)
        
        print(self.list_splits)
        
        self.n_splits   = len(self.list_splits)
        self.topologies = np.zeros((self.n_splits, self.n_topo))

        return
    
    
    def get_all_segpoints(self):
        """
        From the BHV subspace computed, search for all the extremity points of the segments, 
        both in the current tree position and in the supplementary splitting segments of the other topologies.
        Also returns the activation mask per topology.
        
        Parameters
        ----------
        None

        Returns
        -------
        points            : numpy ndarray
            All the extremity points of the segments, in the current tree position and in the supplementary splitting segments. 

        activation_points : numpy ndarray
            Numpy ndarray of size (number of points x number of topology). 
            activation_points[i,j] = 1 if the ith points belongs to the jth topology.
        """

        d = self.position.curve.data.shape[1]

        # 2*n_extremities - 2 + number of supplementary splits
        n_extremities = len(self.position.get_leaves_labels())+1

        n_not_in_position = 0
        for i,split in enumerate(self.list_splits):
            smallest_embedding_node = self.position.find_smallest_split(split)
            if set(smallest_embedding_node.get_leaves_labels())!=split: #Then it is not a split ot the current position tree
                n_not_in_position+=1

        n_seg_points = 2*n_extremities -2 + n_not_in_position #+ 1 + self.n_splits #2*n_extremities - 2  + self.n_splits - (2*n_extremities - 3 - n_extremities)

        seg_points = np.zeros( (n_seg_points, d) )

        n_filled, activation_points = self.fill_points(self.position,seg_points)

        #print(self.n_splits)
        #print(n_filled)
        #print_recursive(self.position)

        for split in self.list_splits:
            
            smallest_embedding_node = self.position.find_smallest_split(split)

            #print("Comparison : ", smallest_embedding_node.get_leaves_labels(), split)

            if set(smallest_embedding_node.get_leaves_labels())!=split: #Then it is not a split of the current position tree
                #print("Different \n")
                seg_points[n_filled,:] = smallest_embedding_node.curve.data[-1,:]

                for j,tree_topo in enumerate(self.unique_topos):
                    smallest_embedding_node = tree_topo.find_smallest_split(split)
                    if set(smallest_embedding_node.get_leaves_labels())==split: #then the topology contains the node
                        activation_points[n_filled,j] = 1
                    """    print(j, '\n')
                    else:
                        print(j, " : ", set(smallest_embedding_node.get_leaves_labels()), " , ", split )"""

                n_filled += 1

        #WARNING : c'est bizarre de devoir refaire les connections de toutes les branches pour une autre topologie ? 

        return  seg_points, activation_points


    def fill_points(self, tree, points, activation_points = None, n = 0):
        """
        Given a list of trees, and a current position in the orthants, compute the points forming all the possible segments of the current topology.
        Activates these points in a mask for all the topologies containing them.
        
        """    

        #Here we must list all the unique segments points from the current position, and add the points of the supplementary splits.

        #print(n)
        if (points==0).all() and n==0: #then we have to start from the root
            if tree.node_id!=0:
                root = self.position.get_root()
                self.fill_points(root,points)
            else: #Ok, start from the root
                activation_points = np.zeros((points.shape[0],self.n_topo))
                activation_points[:2,:] = 1
                points[n,:] = tree.curve.data[0,:]
                n+=1
                
        if n > 1: #Update the activation mask for the point by checking the topologies containing the split
            for i,tree_topo in enumerate(self.unique_topos):
                split = set(tree.get_leaves_labels())
                smallest_embedding_node = tree_topo.find_smallest_split(split)
                if set(smallest_embedding_node.get_leaves_labels())==split: #then the topology contains the node
                    activation_points[n,i] = 1

        points[n,:] = tree.curve.data[-1,:]

        if tree.children==[]:
            activation_points[n,:] = 1

        n+=1

        for child in tree.children: #continue in the children
            n, activation_points = self.fill_points(child, points, activation_points, n)

        return n, activation_points
    
    
    def compute_connections(self, tree, connections = [], ind_parent = 0):
        """

        Parameters
        ----------
        None

        Returns
        -------
        
        """

        if tree.parent is not None and connections == []:
            root = tree.get_root()
            self.compute_connections(root)
        elif tree.parent is None and connections == []:
            connections.append([0,1])
            for child in tree.children:
                self.compute_connections(child, connections, ind_parent=1)
        else:
            split = tree.get_leaves_labels()
            n_point = self.get_split_ind_in_points(split)

            parent = tree.parent
            p_split = parent.get_leaves_labels()
            if parent != tree.get_root():
                n_parent =  self.get_split_ind_in_points(p_split)


            if ind_parent == n_point:
                print("Ok, problem here...", ind_parent, ",", n_point, " , ", n_parent)
                node = self.position.find_smallest_split(split)
                node_split = node.get_leaves_labels()
                print("The current node labels : ", node_split)
                print("Parent leaves : ",tree.parent.get_leaves_labels(), " and current split : ", split)

            connections.append([ind_parent,n_point])
            for child in tree.children:
                self.compute_connections(child, connections, ind_parent=n_point)

        return connections


    def find_equivalent_split(self, split):
        """

        Parameters
        ----------
        None

        Returns
        ------- 
        
        """
    
        equivalent_split = []
        l = np.inf
        ind_selected = None

        for i,split_to_test in enumerate(self.list_splits):
            if set(split).issubset(split_to_test) and len(split_to_test)<l:
                l=len(split_to_test)
                equivalent_split = split_to_test
                ind_selected = i

        if ind_selected is None:
            print("Something wrong... we should have a parent split")

        return ind_selected, equivalent_split
    

    def get_split_ind_in_points(self,split):
        """
        
        """
        node = self.position.find_smallest_split(split)
        node_split = node.get_leaves_labels()

        if set(node_split) == set(split): #then the current split is part of the current subtree
            n_point = node.get_position(node.node_id)+1 #+1 because we count the points, not the segments !!
            
        else: #means that the current split is not in the position tree in the orthants
            ind_equivalent, __ = self.find_equivalent_split(split)

            n_in_position = 0 #we have to check the number of unique splits in the current position before the split of interest
            for split_test in self.list_splits[:ind_equivalent]:
                smallest_embedding_node = self.position.find_smallest_split(split_test)
                if set(smallest_embedding_node.get_leaves_labels())==split_test: #Then it is a split of the current position tree
                    n_in_position+=1
            n_point = 2*(len(self.leaves)+1)-2 + ind_equivalent - n_in_position #

        return n_point


    def compute_all_connections(self):
        """
        
        """
        
        all_con = []

        list_con_tree = []

        for tree in self.unique_topos:
            con_tree = self.compute_connections(tree, connections = [], ind_parent = 0)
            list_con_tree.append(con_tree)
            for con in con_tree: #TODO : Plus efficace en faisant un unique sur l'ensemble des listes...
                if con not in all_con:
                    all_con.append(con)
            #print("For tree : ", i, " the connections are : \n", con, "\n\n\n\n")

        mask_segments = np.zeros( (len(all_con), len(self.unique_topos)), dtype = bool)
        all_connections = np.zeros( (len(all_con) , 2), dtype = int)

        for i,con in enumerate(all_con):
            all_connections[i,0] = con[0]
            all_connections[i,1] = con[1]
            for j,con_tree in enumerate(list_con_tree):
                if con in con_tree:
                    mask_segments[i,j] = 1

        return all_connections, mask_segments
    
