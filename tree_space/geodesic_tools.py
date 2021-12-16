# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 08:36:19 2021
"""

from phylogenetic_distance import bipartite_graph, get_common_edges, get_incidence_matrix, common_edge_contribution, get_nodes_attributes_norm, get_leaves_pairs
from VascularTree_class import VascularTree, plot_recursive, print_recursive
from RatioSequence import RatioSequence, Ratio
import numpy as np

import matplotlib.pyplot as plt
import copy

import math

"""
The class Tree from which one can compute the geodesic distance between two instances requires these methods : 
    
    Tree.get_node(index)
    Tree.descendants()
    Tree.get_nodes_list()
    Tree.get_leaves_labels()
    
And it requires these attributes :
    children  : list of Trees instances
    parent    : a Tree instance
    attribute : a numpy ndarray, containing the coordiantes of one edge (e.g. a length, a set of points...)
    node_id   : integer, the index of the node self in the global Tree.
    
    
"""



class Geodesic(object):
    
    def __init__(self, RS):
        
        assert isinstance(RS, RatioSequence), "Wrong type for geodesic input ({0}), should be a RatioSequence".format(type(RS))
        self.RS = RS
        
        self.common_edges1 = []
        self.common_edges2 = []
        
        self.leaves_contribution_squared = 0
        
    
    def set_ratio_sequence(self, RS):
        self.RS = RS
        return
    
    
    def get_RS(self): return self.RS
    
    def set_common_edges(self, common_edges_pairs):
        """
        
        Parameters
        ----------
        common_edges_pairs : list
            The list of nodes ID, that share in start tree and source tree the same splits.
        
        Returns
        -------
        geodesic_matrix : numpy array
            (n_nodes x n_nodes) matrix containing the geodesic distances between 
            the pairs of nodes in the tree.
        """
        
        n_com = len(common_edges_pairs)
        com1 = [0 for i in range(n_com)]
        com2 = [0 for i in range(n_com)]
        
        for i,p in enumerate(common_edges_pairs):
            com1[i] = p[0]
            com2[i] = p[1]
        
        self.set_common_edges1(com1)
        self.set_common_edges2(com2)
        
        return
    
    
    def set_common_edges1(self, common_edges_1):
        self.common_edges1 = common_edges_1
        return
    
    
    def set_common_edges2(self, common_edges_2):
        self.common_edges2 = common_edges_2
        return
    
    
    def set_leaf_contribution_squared(self, leaf_contrib):
        self.leaves_contribution_squared = leaf_contrib
        return


    def common_edges_contribution(self, Tree1, Tree2):
        
        contrib = 0
        
        for i, e in enumerate(self.common_edges1):
            pair = [e,self.common_edges2[i]]
            c = common_edge_contribution(Tree1, Tree2, pair)
            contrib+=c**2
            
        return np.sqrt(contrib)
    
    
    def get_dist(self, Tree1, Tree2, verbose = False):
        
        RS_sorted = self.RS.getNonDesRSWithMinDist(Tree1, Tree2)
        RS_dist_sqrd   = (RS_sorted.get_distance(Tree1, Tree2))**2
        
        if verbose:
            print("Leaves : ", self.leaves_contribution_squared)
            print("Common : ", self.common_edges_contribution(Tree1, Tree2))
        
        return np.sqrt( self.common_edges_contribution(Tree1, Tree2) + self.leaves_contribution_squared +  RS_dist_sqrd)
    

#END OF CLASS Geodesic


def Geodesic_No_Common_Edges(T1, T2, nodes_of_interest1, nodes_of_interest2):
    """
    
    Note d'implémentation :  dans le code Java, M. Owen implémente les arbres comme 
    des ensembles de branches indépendants les uns des autres, et donc les arbres 
    n'ayant pas de common edges sont des copies des anciens arbres pour lesquels 
    on a enlevé les parties communes. 
    
    J'ai pris le parti de ne pas faire ça et de plutôt lister les branches formant 
    ces arbres sans parties communes. En effet on ne touche pas réellement aux branches 
    dans le calcul de la géodésique, et pour les gros arbres je me dis qu'il n'y a 
    pas besoin de tout copier plusieurs fois.
    
    
    Implementation note: in the Java code, M. Owen implements trees as 
    independent sets of branches, and so trees with no common edges are copies 
    of the old trees for which the common parts have been common parts have been removed. 
    
    Instead I decided to list the branches forming 
    these trees without common edges. Indeed we don't really touch the branches 
    in the calculation of the geodesic, and for the big trees I think that there is no need to 
    need to copy everything several times.
    
    """
        
    assert isinstance(T1,VascularTree) and isinstance(T2,VascularTree), "The arguments must be Tree instances, and not {0}, {1}.".format(type(T1), type(T2))
    
    NNodes1       = len(nodes_of_interest1)
    NNodes2       = len(nodes_of_interest2)
    RS            = RatioSequence()
    ratio_list    = []
    
    if NNodes1 == 0 or NNodes2 == 0:
        print("At least one of the trees has no split, returning 1")
        return 1
    
    ratio_init = Ratio(nodes_of_interest1, nodes_of_interest2, T1, T2)
        
    if NNodes1 == 1 or NNodes2 == 1:
        RS.ratio_list.append(ratio_init)
        return RS
    
    common_edges_list, compatible1, compatible2, splits1, splits2 = get_common_edges(T1, T2)
        
    split_OI1 = {k : splits1[k] for k in nodes_of_interest1}
    split_OI2 = {k : splits2[k] for k in nodes_of_interest2}
        
    assert common_edges_list != [], "The trees are supposed to have no common edge."
    
    IncidMat  = get_incidence_matrix(split_OI1, split_OI2)     # the splits that are neither disjoint, neither including one another.
    BipGraph  = bipartite_graph(IncidMat, get_nodes_attributes_norm(T1,nodes_of_interest1), get_nodes_attributes_norm(T2,nodes_of_interest2))
    
    ratio_list = [ratio_init]
    
    while(len(ratio_list)>0):
        
        ratio_tmp = ratio_list.pop(0)  
        
        #############################################
        #Positions of the nodes selected in the ratio in the lists nodes_of_interest 1 and 2
        n_tmp1 = len(ratio_tmp.edges1)
        aVertices = [0 for i in range(n_tmp1)]
        
        for j, id_ratio in enumerate(ratio_tmp.edges1):
            for i, _id in enumerate(nodes_of_interest1):
                if id_ratio == _id:
                    aVertices[j] = i
                    break 
              
        n_tmp2 = len(ratio_tmp.edges2)
        bVertices = [0 for i in range(n_tmp2)]
        
        for j, id_ratio in enumerate(ratio_tmp.edges2):
            for i, _id in enumerate(nodes_of_interest2):
                if id_ratio == _id:
                    bVertices[j] = i
                    break
        
        ############################################
        
        cover = BipGraph(aVertices, bVertices)
    
        if cover[0,0] == 0 or cover[0,0] == len(aVertices): #Cover is trivial
            RS.ratio_list.append(ratio_tmp) #add to geodesic
        else:
            Ratio1 = Ratio()
            Ratio2 = Ratio()
            
            select1 = sum(cover[2,:]!=-1)
            select2 = sum(cover[3,:]!=-1)
                        
            j = 0
            #Split the ratio based on the cover
            for i in range(len(aVertices)):
                if j < select1 and aVertices[i] == cover[2,j] :
                    Ratio1.edges1.append(nodes_of_interest1[aVertices[i]])
                    j+=1
                else: #the split is not in the cover, it is dropped first
                    Ratio2.edges1.append(nodes_of_interest1[aVertices[i]])
            
            j = 0
            for i in range(len(bVertices)):
                if j < select2 and bVertices[i] == cover[3,j] :
                    Ratio2.edges2.append(nodes_of_interest2[bVertices[i]])
                    j+=1
                else: #the split is not in the cover, it is dropped first
                    Ratio1.edges2.append(nodes_of_interest2[bVertices[i]])
            
            Ratio1.update_lengths(T1, T2)
            Ratio2.update_lengths(T1, T2)
            
            ratio_list = [Ratio1, Ratio2] + ratio_list

    return RS


def grow_tree(T, node2insert, new_id = -1, orthant_time = 1):
    """
    Add a node to a tree and make it grow.
    The method is exactly the same as shrinking the source branch, but starting from the 
    target one with a time t2 = 1-orthant_time
    
    """
    
    target_leaves_labels = node2insert.get_leaves_labels()
    
    source_parent = T.find_smallest_split(target_leaves_labels)

    children2move = []
    for child in source_parent.children:
        tmp_leaves = child.get_leaves_labels()
        if set(tmp_leaves).issubset(target_leaves_labels) or child.label in target_leaves_labels:
            children2move.append(child.node_id)
    
    node2insert.children = []
    list_ids = T.get_nodes_list([])

    if new_id not in list_ids:
        node2insert.node_id = new_id
    elif len(list_ids) not in list_ids:
        node2insert.node_id = len(list_ids)
    else:
        node2insert.node_id = -len(list_ids)

    target_data = np.tile(node2insert.curve.data[0,:], [node2insert.curve.data.shape[0],1])
    node2insert.interpolate2target(target_data, 1-orthant_time) #here we are moving from the shrinked branch to the target tree branch
    source_parent.insert_node(node2insert, children2move)
        
    return T


#%% The plot of the geodesic
def follow_geodesic(geod, tree, tree_target, show = True):
    
    cpt = -1
    for r in geod.get_RS().ratio_list:
        
        for edge_id in r.edges1:
            node = tree.get_node(edge_id)
            node.shorten(node.length-1, keep_size = True)
            tree.remove_node(node.node_id, node, update = False)
            if show:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                plot_recursive(tree,ax)    
        
        #Inserting a branch
        for id2insert in r.edges2:   
            node2insert   = copy.deepcopy(tree_target.get_node(id2insert))
            #print('Node to insert split : ', node2insert.get_leaves_labels())
            tree = grow_tree(tree,node2insert, new_id = cpt)
            cpt-=1
            if show:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                plot_recursive(tree,ax)
                
    return 


#%% The plot of the geodesic
def follow_geodesic_interpolation(geod, tree, tree_target, target_time = 1):
    """
    Given a geodesic in the space of phylogenetic trees, the start and end points 
    in this space, compute the point at time 'target_time' along this 
    geodesic. As the 

    Parameters
    ----------
    geod        : Geodesic object
        The pre-computed geodesic between tree and tree_target defining which orthant to 
        walk through along the path. 
    tree        : Tree
        The source tree.
    tree_target : Tree
        The target Tree.
    target_time : float, optional
        The time to stop in the geodesic path. The default is 1.

    Returns
    -------
    new_tree : Tree
        The geodesic interpolation at time `target_time` between tree and tree_target.

    """
    
    assert target_time>=0 and target_time<=1, "target_time should be between 0 and 1."

    if target_time == 1: #then target_time=1
        return copy.deepcopy(tree_target)

    common_edges_list, compatible1, compatible2, splits1, splits2 = get_common_edges(tree, tree_target)
    leaves_pairs_ids = get_leaves_pairs(tree, tree_target)

    new_tree = copy.deepcopy(tree)
    cpt = -1
    
    s_root = new_tree.get_root()
    t_root = tree_target.get_root()
    s_root.interpolate2target(t_root.curve.data,target_time)
            
    for ep in common_edges_list:
        source_edge = new_tree.get_node(ep[0])
        target_edge = tree_target.get_node(ep[1])
        source_edge.interpolate2target(target_edge.curve.data, target_time, graft = True)
    
    #Then deal with the common parts
    for lp in leaves_pairs_ids:
        source_leaf = new_tree.get_node(lp[0])
        target_leaf = tree_target.get_node(lp[1])
        source_leaf.interpolate2target(target_leaf.curve.data, target_time, graft = True)
    
    """for i,r in enumerate(geod.get_RS().ratio_list):

        for ind in r.edges1:
            print("Source split : ",tree.get_node(ind).get_leaves_labels())

    for i,r in enumerate(geod.get_RS().ratio_list):

        for ind in r.edges2:
            print("Target split : ",tree_target.get_node(ind).get_leaves_labels()) """      

    for i,r in enumerate(geod.get_RS().ratio_list):
        
        ratio_time = r.get_time(tree, tree_target)

        if target_time>=ratio_time: #If we are in the ending orthant, compute the length ratio, completely shrink and remove otherwise.

            for edge_id in r.edges1:
                
                node = new_tree.get_node(edge_id)
                target_data = np.tile(node.curve.data[0,:], [node.curve.data.shape[0],1])

                orthant_time = 1
                node.interpolate2target(target_data, orthant_time)
                if target_time!=ratio_time:
                    new_tree.remove_node(node.node_id, node, update = False, offset=True, pop=False)
                else:
                    # do not remove : we are at the boundary
                    print("WE REACHED A BOUNDARY... WHAT TO DO ?")
                #new_tree.remove_node(node.node_id, node, update = False, offset=True, pop=False)

            if target_time!=ratio_time:
                for id2insert in r.edges2:
                    node2insert = copy.deepcopy(tree_target.get_node(id2insert))
                    cpt-=1
                    
                    orthant_time = 0
                    if i < len(geod.get_RS().ratio_list)-1:
                        next_time = (geod.get_RS().ratio_list[i+1]).get_time(tree, tree_target)
                        if target_time >= next_time:
                            orthant_time = 1
                        else:
                            orthant_time = (target_time-ratio_time)/abs( next_time - ratio_time)
                    elif i == len(geod.get_RS().ratio_list)-1:
                        orthant_time = (target_time-ratio_time)/abs(1 - ratio_time)
                    
                    new_tree = grow_tree(new_tree.get_root(), node2insert, new_id = cpt, orthant_time=orthant_time)

        else: #target_time<ratio_time

            #We are after the target time, we only have to shrink the edges of the source tree in this ratio
            orthant_time = 0
            previous_time = 0
            """if i > 0:
                previous_time = (geod.get_RS().ratio_list[i-1]).get_time(tree, tree_target)
                orthant_time = (target_time-previous_time)/abs( ratio_time - previous_time )
            elif i == 0:"""
            orthant_time = target_time/abs(ratio_time)

            if i == 0 or previous_time <=target_time: #then we are in the current orthant and we should shrink source branches
                for edge_id in r.edges1:
                    
                    node = new_tree.get_node(edge_id)
                    target_data = np.tile(node.curve.data[0,:], [node.curve.data.shape[0],1])
                    node.interpolate2target(target_data, orthant_time)

        """if ratio_time == 1: #these are nodes to remove
            for edge_id in r.edges1:
                
                node = new_tree.get_node(edge_id)
                target_data = np.tile(node.curve.data[0,:], [node.curve.data.shape[0],1])

                orthant_time = 1
                node.interpolate2target(target_data, orthant_time)
                new_tree.remove_node(node.node_id, node, update = False, offset=True, pop=False)"""

    new_tree.reset_ids(0) #clean_ids(force = True) #
  
    return new_tree.get_root()

