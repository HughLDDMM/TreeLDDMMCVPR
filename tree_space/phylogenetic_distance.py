# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 14:37:20 2021
"""

#from VascularTree_class import VascularTree
import numpy as np
import copy
    

TOLERANCE = 0.00000001

#Functions to compute the attribute norm squared
def sqeuclidean(x): return np.inner(x, x)


def edge_average(T, edge_list):
    """
    Given a VascularTree and a list of edges of interest, will compute the "norm" of 
    this set of edges.
    WARNING : This depend on the attribute of the nodes.

    #TODO : a fusionner avec get_nodes_attributes_norm

    Parameters
    ----------
    T : VascularTree
        The VascularTree from which we selected edges.
    edge_list : list
        List of edges indices.

    Returns
    -------
    float
        The weight of a collection of edges.

    """

    d = 0
    
    for ind in edge_list:
        node = T.get_node(ind)
        if len(node.attribute.shape)>1:
            #tmp = np.ndarray.flatten(node.attribute)
            tmp = node.curve.length
        else:
            tmp = node.attribute
        d += sqeuclidean(tmp)
        
    return np.sqrt(d)


def common_edge_contribution(T1, T2, edge_pair):
    """
    Given a pair of trees and a pair of indices corresponding to a common edge
    between the trees, returns this edge contribution to the geodesic.

    Parameters
    ----------
    T1        : VascularTree
        The first VascularTree from which we selected edges.
    T2        : VascularTree
        The second VascularTree from which we selected edges.
    edge_pair : list
        List of 2 indices.

    Returns
    -------
    float
        The weight of a collection of edges.

    """
    
    assert len(edge_pair)>=1, 'No pair given to compute the common edge contibution !'
        
    d = 0
    
    node1 = T1.get_node(edge_pair[0])
    node2 = T2.get_node(edge_pair[1])
    
    diff = node1.attribute - node2.attribute
    
    if len(diff.shape)>1:
        tmp = np.ndarray.flatten(diff)
    else:
        tmp = diff
    d += sqeuclidean(tmp)
    
    return np.abs(d)



def get_leaves_contributions(Tree1, Tree2):
    """
    Given a pair of trees, compute the contribution of the leaves to the geodesic distance,
    a.k.a the sum of the squared norms of the difference of the attributes.

    Parameters
    ----------
    T1        : VascularTree
        The first Tree from which we selected edges.
    T2        : VascularTree
        The second Tree from which we selected edges.
    edge_pair : list
        List of 2 indices.

    Returns
    -------
    float
        The weight of a collection of edges.
    """
    leaves1 = Tree1.get_leaves_IDs()
    leaves2 = Tree2.get_leaves_IDs()

    leaves_contribution_squared = 0
    
    leaves_2_paired = [0 for l in leaves2]
    
    for i in leaves1:
        leaf1 = Tree1.get_node(i)
        
        paired = False
        for ind_j, j in enumerate(leaves2):
            leaf2 = Tree2.get_node(j)
            if leaf2.label == leaf1.label:
                paired = True
                diff = leaf1.attribute - leaf2.attribute
                tmp = np.ndarray.flatten(diff)
                leaves_contribution_squared += sqeuclidean(tmp)
                leaves_2_paired[ind_j] = 1
                break
    
        if not paired: #Then leaf 1 at iteration i has no corresponding leaf in Tree 2 
            print("WARNING, did not find a leaf in target VascularTree corresponding to one in source VascularTree, is that normal?")
            tmp = np.ndarray.flatten(leaf1.attribute)
            leaves_contribution_squared += sqeuclidean(tmp)
    
    #Then dealing with leaves with no correspondence in Tree2
    for k,test in enumerate(leaves_2_paired):
        if not test:
            print("WARNING, did not find a leaf in source VascularTree corresponding to one in target VascularTree, is that normal?")
            leaf2 = Tree2.get_node(leaves2[k])
            tmp = np.ndarray.flatten(leaf2.attribute)
            leaves_contribution_squared += sqeuclidean(tmp)
    
    return leaves_contribution_squared


def get_nodes_attributes_norm(T, edges_of_interest):
    
    norm_list = [0 for i in edges_of_interest]
    
    for i, ind in enumerate(edges_of_interest):
        node = T.get_node(ind)
        if len(node.attribute.shape)>1:
            tmp = np.ndarray.flatten(node.attribute)
        else:
            tmp = node.attribute
        norm_list[i] = np.sqrt(sqeuclidean(tmp))
    
    return norm_list


def get_splits(tree_root, intern = True):
    """
    Walk through all the tree nodes and returns the splits in a python dict.

    Parameters
    ----------
    tree_root : VascularTree
        The starting node to search for splits.
    intern    : bool
        If False, also return the splits for the leaves.

    Returns
    -------
    splits    : dict.

    """
    
    splits_dic = {}
    
    for node in tree_root.descendants():
        
        if not intern:
            splits_dic[node.node_id] = set(node.get_leaves_labels())
            
        elif node.parent is not None and node.children != []:
            splits_dic[node.node_id]=node.get_leaves_labels()
    
    return splits_dic


def crosses(split1, split2):
    """
    Check whether the splits intersect.
    
    """
    intersect = False
    #print(split1, split2)
    #if bool(set(split1).intersection(split2)):
    if not (set(split1).isdisjoint(split2) or set(split1).issubset(split2) or set(split2).issubset(split1)):
        intersect = True
    
    return intersect
    

def is_split_compatible(split2test, all_splits_target):
    """
    A split is said compatible if neither of the sets of leaves they define have 
    an intersection with the sets defined in the target trees. 

    Parameters
    ----------
    split2test        : list
        List of leaves descendant from a given node.
    all_splits_target : dict
        Python dictionnary containing all the splits of a given Tree.

    Returns
    -------
    is_compatible : bool
        True if a split is compatible with target splits. 
        (see M. Owen 2011, A Fast Algorithm for Computing Geodesic
Distances in Tree Space).

    """
    
    
    is_compatible = True
    
    for k in all_splits_target.keys():
        
        if crosses(split2test, all_splits_target[k]):
            is_compatible = False
            return is_compatible
    
    return is_compatible


def get_incidence_matrix(splits1, splits2):
    
    n = len(splits1.keys())
    m = len(splits2.keys())
    
    IncidenceMatrix = np.zeros((n, m), dtype=bool)
    
    for i,s1 in enumerate(splits1.values()):
        for j,s2 in enumerate(splits2.values()):
            IncidenceMatrix[i,j] = crosses(s1, s2)
    
    return IncidenceMatrix


def get_common_edges(Tree1, Tree2):
    """
    Get the interior edges of the trees splitting them into the same sets of 
    leaves.
    
    
    Parameters
    ----------
    Tree1 : VascularTree
        First tree to look in.
    Tree2 : VascularTree
        Second tree to look in.

    Returns
    -------
    common_edges_list : list
        A list containing pairs of nodes id [node_tree1_id, node_tree2_id] for
        all pairs of common edges.
    splits1  : dict.
        The splits of the first tree.
    splits2  : dict.
        The splits of the second tree.
        
    """
    
    #print("WARNING : Right now the common edges list only contains the real common edges, when JAVA code also adds compatible edges")
    
    common_edges_list = []
    compatible1 = []
    compatible2 = []
    
    leaves1 = Tree1.get_leaves_labels()
    leaves2 = Tree2.get_leaves_labels()
    
    if set(leaves1)!=set(leaves2):
        print("The two trees have different sets of leaves.")
        print("Leaves in Tree1 : {0}".format(leaves1))
        print("Leaves in Tree2 : {0}".format(leaves2))
        print("Returning 1")
        return 1
    
    splits1 = get_splits(Tree1)
    splits2 = get_splits(Tree2)
    
    for k1 in splits1.keys():
        S_k1 = set(splits1[k1])
        
        sets_splits2 = [set(s) for s in splits2.values()]

        if S_k1 in sets_splits2 :  #Then there is a common edge
            ind = list(sets_splits2).index(S_k1)
            k2 = list(splits2.keys())[ind]
            common_edges_list.append([k1, k2])
            
        elif is_split_compatible(S_k1,splits2):
            compatible1.append(k1)
            
    for k2 in splits2.keys():
        S_k2 = splits2[k2]
        if is_split_compatible(S_k2,splits1):
            compatible2.append(k2)

    """print("\n\n\n\n\n\n")
    print(common_edges_list)
    print("\n\n\n\n\n\n")"""

    return common_edges_list, compatible1, compatible2, splits1, splits2


def check_same_topology(Tree1, Tree2):
    
    check = False
    
    common_edges_list, compatible1, compatible2, splits1, splits2 = get_common_edges(Tree1, Tree2)
    
    n_extremities = len(Tree1.get_leaves_IDs())+1
    n_segments    = 2*n_extremities - 3
    n_inner       = n_segments - n_extremities

    #print("WARNING : not sure about check criterion !!!!!!!!!!!!!!!!!")
    if len(common_edges_list)==n_inner:
        check=True
    
    return check


def get_leaves_pairs(Tree1, Tree2):
    
    source_leaves = Tree1.get_leaves()
    target_leaves = Tree2.get_leaves()
    
    leaves_pairs = []
    
    for sl in source_leaves:
        tmp = [sl.node_id]
        for tl in target_leaves:
            if sl.label == tl.label:
                tmp.append(tl.node_id)
                
        if len(tmp) != 2:
            print("WARNING, this is not a pair.")
        leaves_pairs.append(tmp)
    
    return leaves_pairs


def split_on_common(Tree1, Tree2):
    """
    

    Parameters
    ----------
    Tree1 : VascularTree
        DESCRIPTION.
    Tree2 : VascularTree
        DESCRIPTION.

    Returns
    -------
    list_pairs_nocommon : list 
        List of pairs of trees with no common edges.

    """
    
    list_pairs_nocommon = []
    
    def next_split(t1, t2):
        """
        Parameters
        ----------
        t1 : VascularTree
            DESCRIPTION.
        t2 : VascularTree
            DESCRIPTION.
    
        Returns
        -------
        None
        """
                
        common_edges_list, compatible1, compatible2, splits1, splits2 = get_common_edges(t1, t2)
        
        if splits1=={} or splits2=={}:
            return
        
        if len(common_edges_list)==0:
            list_pairs_nocommon.append([t1.get_interior_nodes_list(),t2.get_interior_nodes_list()])
            return


        common_edge_1 = t1.get_node(common_edges_list[0][0])
        common_edge_2 = t2.get_node(common_edges_list[0][1])
                
        At1 = copy.deepcopy(t1)
                
        At1.remove_node(common_edges_list[0][0], merge = True, update = False)
        Bt1 = copy.deepcopy(common_edge_1)
        Bt1.parent = None
        
        At2 = copy.deepcopy(t2)
        At2.remove_node(common_edges_list[0][1], merge = True, update = False)
        Bt2 = copy.deepcopy(common_edge_2)
        Bt2.parent = None
        
        next_split(At1, At2)
        next_split(Bt1, Bt2)
        
        return
    
    next_split(Tree1, Tree2)
    
    return list_pairs_nocommon


def bipartite_graph(incidence_mat, tree1_attributes_norms, tree2_attributes_norms):
    """
    

    Parameters
    ----------
    incidence_mat : np.ndarray (bool)
        The boolean matrix whose coordinates (i,j) is 1 of node i of VascularTree 1 doesn't cross node j of VascularTree 2.
    tree1_attributes_norms : list
        The list containing each node's attribute norm for VascularTree 1.
    tree2_attributes_norms : list
        The list containing each node's attribute norm for VascularTree 2.

    Returns
    -------
    vertex_cover : function
        Given two sets of vertices in source and target trees, returns the cover,
        the min-normalized_square-weighted vertex cover.

    """
    
    edge = incidence_mat
    
    n1 = len(tree1_attributes_norms)
    n2 = len(tree2_attributes_norms)
    
    n = max(n1,n2)
        
    V1 = np.zeros((n,4)) #weights, residual, label, pred 
    V2 = np.zeros((n,4)) #weights, residual, label, pred 
    
    V1[:n1,0]=np.asarray(tree1_attributes_norms)
    V2[:n2,0]=np.asarray(tree2_attributes_norms)
        
    #print("Not sure about the structure of the vertex... should create a class to have more explicit .weight, .residual, .label and .pred?")
    
    def vertex_cover(ind_nodes_1, ind_nodes_2):
                
        nVC1 = len(ind_nodes_1)
        nVC2 = len(ind_nodes_2)
        
        flow12 = np.zeros((n1,n2))
        tot = 0
        augmentingPathEnd = -1
        
        ScanList1 = [0 for i in range(n1)]
        ScanList2 = [0 for i in range(n2)]
        
        cover = np.zeros((4,n))-1
        
        #Normalize the weights:
        for i in range(nVC1): 
            tot += V1[ind_nodes_1[i], 0]
        if tot != 0:
            for i in range(nVC1): 
                V1[ind_nodes_1[i], 1] = V1[ind_nodes_1[i], 0]/tot
        tot = 0
        for j in range(nVC2): 
            tot += V2[ind_nodes_2[j], 0]
        
        if tot != 0:
            for j in range(nVC2): 
                V2[ind_nodes_2[j], 1] = V2[ind_nodes_2[j], 0]/tot
        tot = 1
        
        while tot > 0:
                        
            tot = 0
            for i in range(nVC1):
                V1[ind_nodes_1[i], 2] = -1
                V1[ind_nodes_1[i], 3] = -1
                
            for j in range(nVC2):
                V2[ind_nodes_2[j], 2] = -1
                V2[ind_nodes_2[j], 3] = -1
                
            ScanListSize1 = 0
            
            for i in range(nVC1):
                if (V1[ind_nodes_1[i], 1]> TOLERANCE):
                    V1[ind_nodes_1[i], 2] = V1[ind_nodes_1[i], 1]
                    ScanList1[ScanListSize1]=ind_nodes_1[i] 
                    ScanListSize1+=1
                #Lignes commentées car par la boucle for précédente, V1[ind_nodes_1[i], 2] = -1 
                #else:
                #    V1[ind_nodes_1[i], 2]=-1;
            
            for i in range(nVC2): V2[i, 2] = -1 #Not sure about this line... strange to directly modify with index i here...
            
            test_break = False
            
            while( (ScanListSize1!=0) and (not test_break)):
                
				#Scan the A side nodes
                ScanListSize2 = 0
                for i in range(ScanListSize1):
                    for j in range(nVC2):
                        if (edge[ScanList1[i],ind_nodes_2[j]] and V2[ind_nodes_2[j], 2]==-1): 
                            V2[ind_nodes_2[j], 2] = V1[ScanList1[i], 2] 
                            V2[ind_nodes_2[j], 3] = ScanList1[i]
                            ScanList2[ScanListSize2]=ind_nodes_2[j]
                            ScanListSize2+=1
						
				# Scan the B side nodes
                ScanListSize1 = 0

                for j in range(ScanListSize2): 
                    if (V2[ScanList2[j], 1] > TOLERANCE): 
                        tot=min(V2[ScanList2[j], 1], V2[ScanList2[j], 2])
                        augmentingPathEnd=ScanList2[j]
                        test_break = True
                        break
					
                    else:
                        for i in range(nVC1):
                            if (edge[ind_nodes_1[i],ScanList2[j]] 
                                and V1[ind_nodes_1[i], 2]==-1 
                                and flow12[ind_nodes_1[i],ScanList2[j]]>0):
                                
                                V1[ind_nodes_1[i], 2] = min(V2[ScanList2[j], 2],flow12[ind_nodes_1[i],ScanList2[j]])
                                V1[ind_nodes_1[i], 3] = ScanList2[j]
                                ScanList1[ScanListSize1]=ind_nodes_1[i]
                                ScanListSize1+=1
            
            if (tot>0): #flow augmentation 
                V2[augmentingPathEnd, 1]-=tot
                Bpathnode=augmentingPathEnd
                Apathnode=int(V2[Bpathnode, 3])
                flow12[Apathnode, Bpathnode]+=tot
            
                while (V1[Apathnode,3]!=-1):
                    Bpathnode=int(V1[Apathnode, 3])
                    flow12[Apathnode, Bpathnode] -= tot
                    Apathnode=int(V2[Bpathnode, 3])
                    flow12[Apathnode, Bpathnode] += tot
				
                V1[Apathnode, 1]-=tot
                
            else:  #min vertex cover found, unlabeled A's, labeled B's
            
                k=0
                for i in range(nVC1):
                    if (V1[ind_nodes_1[i], 2]==-1):
                        cover[2, k]=ind_nodes_1[i]
                        k+=1
 					
                cover[0,0]=k
                k=0
                for j in range(nVC2):
                    if (V2[ind_nodes_2[j], 2]>= 0):
                        cover[3, k]=ind_nodes_2[j]
                        k+=1
 					
                cover[1,0]=k
						  
        return cover
    
    return vertex_cover
    




