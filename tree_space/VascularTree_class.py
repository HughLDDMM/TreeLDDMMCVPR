# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 17:11:52 2021
"""
from node_class  import Node
from curve_class import Curve

import copy
import numpy as np

import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(100000)

import math 



######################################SOME INPUT/OUTPUT FUNCTIONS (import/export data)#######
from pyvtk import VtkData

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






def BranchPath(Path2BranchesRep,current_branch):
    
    branch_path = Path2BranchesRep+'/branch'
    digits = int(math.log10(current_branch))+1 #number of digits, for the branch path

    for ind_zero in range(4-digits):
        branch_path+='0'    

    branch_path+=str(current_branch)+'.txt'
    
    return branch_path





# THE TREE CLASS
class VascularTree(Node):
    
    """
    
    """
    
    def __init__(self, tree = None, data = None, node_id = -1, ind_first_point = 0, 
                 label = -1, attribute_type = 'length', parent = None, children = None):
        """
        Vascular Tree
        
        Combines the class Curve and Node to create a class of Tree of Curves. 

        Parameters
        ----------
        tree            : VascularTree, optional
            A VascularTree to instantiate from. Default is None.
            
        data            : numpy ndarray, optional
            The points of the branch associated to a given node. Default is None.
            
        node_id         : int, optional
            The node id. Default is -1.
            
        ind_first_point : int, optional
            Index of the first point of the branch in the whole tree. Default is 0.
            
        

        Returns
        -------
        None (the class instance.)

        
        """
        if tree is not None:
            #Create a copy of a Tree instance
            assert isinstance(tree,VascularTree), "The initial node must be a proper class Tree"
            self.__init__(data = copy.deepcopy(tree.curve.data), 
                          node_id = tree.node_id, 
                          ind_first_point = tree.curve.ind_first_point,
                          label = tree.label,
                          attribute_type = tree.attribute_type,
                          parent = copy.deepcopy(tree.parent), 
                          children = copy.deepcopy(tree.children))  

        else: #Create a Tree instance from informations
            
            self.curve = Curve(data = data, ind_first_point = ind_first_point)
            
            #Instanstiate the Node part.
            super().__init__(node_id = node_id, label = label, parent = parent, children = children)
            
            #FOR THE TREE SHAPE SPACE
            self.attribute_type = attribute_type
            if attribute_type == 'length':
                self.attribute = np.array([self.curve.length])
            else:
                self.attribute = self.curve.data
            
            if parent is not None:
                assert isinstance(parent, VascularTree), "Error : the specified parent is not a Tree object"
                self.parent = parent
            else:
                self.parent = None
                
            if self.parent is not None:
                self.depth = self.parent.depth+1
            else:
                self.depth = 0           
                
            #The inactive nodes are set to False, and corresponds to other existing topologies
            #Default is True
            self.active = True


    def __del__(self):
        del self
        
        
    def del_all(self):
        del self.curve.data
        del self.node_connections
        del self.matrix_connections
        for child in self.children:
            child.del_all()
        self.parent.del_all()    


    def update_lengths(self):
        """
        Vascular Tree
        
        The points connections are updated with respect to the parent last indices. 
        Check whether the first point is shared with the parent. If so, take the 
        correct index in the tree. 
        """
        
        root = self.get_root()
        list_nodes = root.get_nodes_list([])
        
        for n in list_nodes:
            root.get_node(n).curve.update_length()
        
        return

    def update_connections(self, n_pts):
        """
        Vascular Tree
        
        The points connections are updated with respect to the parent last indices. 
        Check whether the first point is shared with the parent. If so, take the 
        correct index in the tree. 
        """
        if self.curve.data.shape[0]!=0:
            ind_previous, previous_point = self.get_previous_point(None)
            
            if ind_previous is not None:
                if (previous_point == self.curve.data[0,:]).all():
                    tmp = [[n_pts+j+i for j,c in enumerate(con)] for i,con in enumerate(self.curve.points_connections[1:]) ]
                    self.curve.points_connections = [[ind_previous, n_pts]] + tmp
                    self.curve.ind_first_point = ind_previous
                    n_pts += self.curve.data.shape[0]-1
                    
                else:
                    self.curve.points_connections = [[n_pts+j+i for j,c in enumerate(con)] for i,con in enumerate(self.curve.points_connections) ]
                    self.curve.ind_first_point = n_pts
                    n_pts+= self.curve.data.shape[0]
            else:
                self.curve.points_connections = [[n_pts+j+i for j,c in enumerate(con)] for i,con in enumerate(self.curve.points_connections) ]
                self.curve.ind_first_point = n_pts
                n_pts+= self.curve.data.shape[0]
            
        for child in self.children:
            n_pts = child.update_connections(n_pts)

        return n_pts


    #%% ACCESSORS
    def get_previous_point(self, ind_previous):
        """
        Vascular Tree
        CARE : this will not work with a graph
        
        Parameters
        ----------
        ind_previous : int
            The index of the previous point in the tree.
        
        Returns
        -------
        ind_previous   : int
                         The index of the previous point in the whole tree.
                         None if no previous point in the tree.
        previous_point : numpy ndarray
                         The previous point connected to the current node. 
                         None if no previous point in the tree.

        """

        previous_point = None
        
        if self.parent is not None:
            
            if ind_previous is None: #then we keep looking
            
                if self.parent.curve.data.shape[0] != 0: #then the parent has at least one point:
                    ind_previous = self.parent.curve.points_connections[-1][-1]
                    previous_point = self.parent.curve.data[-1,:]
                else:
                    ind_previous, previous_point = self.parent.get_previous_point(ind_previous)

        return ind_previous, previous_point
        

    def dist_to_ancestor(self, ancestor, dist = 0):
        """
        Vascular Tree
        
        Compute the geodesic distance to a given not (if this node is an ancestor).
        
        Parameters
        ----------
        ancestor : Tree
            The node we want to find the distance to (has to be an ancestor of self).

        Returns
        -------
        dist : float
            The geodesic distance between the two nodes 'self' and 'ancestor'.
            If ancestor is not self's ancestor, return -2
        
        """
        
        if dist == 0:
            if ancestor.node_id not in self.ancestors():
                print("The provided node is not an ancestor, please provide one. \n Returning -2")
                return -2.
        
        if self.parent != ancestor and self.parent is not None:
            
            d_transition = np.linalg.norm(self.curve.data[0,:]-self.parent.curve.data[-1,:])
            
            dist += self.parent.curve.get_length()+d_transition
            dist = self.parent.dist_to_ancestor(ancestor, dist)
            
        if self.parent is None:
            print("WARNING : the provided node is not an ancestor of the starting node, but the first check did not work.")
            return -2
        
        return dist
    
    
    def geodesic_distance(self, target_node):
        """
        Vascular Tree
        
        Compute the geodesic distance inside the tree between self and a target node.

        Parameters
        ----------
        target_node : Tree
            The node we want to find the distance to.

        Returns
        -------
        dist : float
            The geodesic distance between the two nodes 'self' and 'target_node'.

        """
        
        dist = 0
        
        if self == target_node:
            return 0
        
        lca = self.lowest_common_ancestor(target_node)

        if lca == self:
            dist = target_node.dist_to_ancestor(lca, 0)
                
        elif lca == target_node:
            dist = self.dist_to_ancestor(lca, 0)
        
        else:
           dist1 = self.dist_to_ancestor(lca, 0)
           dist2 = target_node.dist_to_ancestor(lca, 0)
           dist = dist1+dist2
        
        return dist
    
    
    def geodesic_matrix(self, list_nodes = [], geodesic_matrix = None):
        """
        Vascular Tree
        
        Compute the matrix of geodesic distances between all pairs of nodes in the list of provided nodes.
        If no list provided, computes between all the pairs of nodes in the tree.

        """
        
        if list_nodes == []:
            root = self.get_root()
            list_nodes = root.get_nodes_list([])
        if geodesic_matrix is None:
            n_nodes = len(list_nodes)
            geodesic_matrix = np.zeros((n_nodes, n_nodes))
                
        for i, node_id_i in enumerate(list_nodes[:-1]):
            
            node_i = self.get_node(node_id_i)
            
            for j, node_id_j in enumerate(list_nodes[i+1:]):
                
                #print('Node : {0}, node : {1}'.format(node_id_i, node_id_j))
                
                node_j = self.get_node(node_id_j)
            
                d = node_i.geodesic_distance(node_j)
                
                ind_j = i+1+j
                
                geodesic_matrix[i,ind_j] = d
                geodesic_matrix[ind_j,i] = d

        return geodesic_matrix
    
    
    def get_max_geodesic_length(self):
        """
        Vascular Tree
        
        Compute the maximum length from the root to the leaves of the tree.

        """
        max_length = 0
        root = self.get_root()
        leaves = root.get_leaves()

        for leaf in leaves:
            l_length = leaf.dist_to_ancestor(root) + leaf.curve.length
            if l_length > max_length:
                max_length = l_length

        return max_length+root.curve.length


    def set_attribute(self, attribute):
        """
        Vascular Tree
        """
        
        if attribute not in ['length', 'curve']:
            print('Attribute not handled right now (accepted : "length" or "curve"), returning')
            return
        
        root = self.get_root()
        
        descendants = root.descendants()
        
        for n in descendants:
            
            if attribute == 'length':
                n.attribute = np.array([n.curve.length])
            else:
                n.attribute_type = 'curve'
                n.attribute = self.curve.data
        
        return
    
    
    #%% ADD/REMOVE
    def add_child(self, node, set_id = False, prepend=False, offset = False):
        """
        Add a child to self.

        Parameters
        ----------
            
        node    : Tree,
            If provided, remove this node directly instead of searching by the 
            node id. 
            
        set_id  : bool, optional (Default : False)
            If True, will check if the new child's id is available, and set 
            a new one if not. 
            
        prepend : bool, optional (Default : False)    
            If True, update the branches ids and connections once the node 
            removed.
            
        offset : bool, optional (Default : False)
            If True, and if merge is True, will translate the subtree to match
            the new parent's last point.
            
        Returns
        -------
        None.

        """
        if self.curve.data is None:
            self.__init__(self, node)
        assert type(node.curve.data) == type(self.curve.data), "The new child must have same data type as the current node {0}".format(self.node_id)
        
        super().add_child(node, set_id = set_id, prepend = prepend)
        
        if offset:
            delta = self.curve.data[-1,:] - node.curve.data[0,:]
            for d in node.descendants():
                d.curve.data += delta
    
        return


    def add_node(self, node, parent_id = 0):
        """
        Add a node to a given position.

        Parameters
        ----------
            
        node       : Tree,
            If provided, remove this node directly instead of searching by the 
            node id. 
            
        parent_id  : int, optional (Default : 0)
            If provided, will check if the parent's id is available, and add the new child to this node. 
            
        Returns
        -------
        None.

        """

        assert isinstance(node,VascularTree), "The new node must be a proper class Tree"
        
        if self.curve.data is None:
            print("Adding a node to an empty Tree, initializing with this node")
            self.__init__(node = node)
        else:
            parent = self.get_node(parent_id)
            if parent is not None:
                #children = copy.deepcopy(parent.children)
                parent.children.append(node)
                node.parent = parent
            else:
                print("Did not find the parent in the current tree (root : {0})".format(self.node_id))
        return
        
        
    def delete_subtree(self, subtree_root_id, subtree_root = None):
        """
        Add a node to a given position.

        """

        if subtree_root is None:
            subtree_root = self.get_node(subtree_root_id)
        
        for child in subtree_root.children:
            child.delete_subtree(child.node_id)
            
        if(subtree_root is None):
            print("Found nothing to remove")
            
        del subtree_root
        
        return
        
    
    def shorten(self, length2shorten, keep_size = False):
        """
        Curve + Vascular Tree
        
        Shorten a current branch.

        Parameters
        ----------
        length2shorten : float
            Length of the branch to remove.

        Returns
        -------
        None.

        """
      
        translation = self.curve.shorten(length2shorten, keep_size = keep_size)
        root = self.get_root()
        root.update_connections(0)
        
        descendants = self.descendants()
        descendants.pop(0)
        
        for node in descendants:
            node.curve.data-=translation

        #self.curve.shorten(length2shorten)

        return
    
    
    def remove_node(self, node_id, node2remove = None, merge = True, update = True, offset = False, pop = True):
        """
        
        WARNING, HERE USING AN OVERWRITTEN METHOD, NEED TO OVERWRITE AS WELL? 
        
        Remove a node in the tree and update the nodes ids. 

        Parameters
        ----------
        node_id     : int
            The id of the node to remove.
            
        node2remove : Tree, optional (Default : None)
            If provided, remove this node directly instead of searching by the 
            node id. The default is None.
            
        merge       : bool, optional (Default : True)
            If True, merge the descendants to the parent's children. 
            
        update      : bool, optional (Default : True)    
            If True, update the branches ids and connections once the node removed.
            
        offset      : bool, optional (Default : False)
            If True, and if merge is True, will translate the subtree to match the new parent's last point.
            
        pop         : bool, optional (Default : True)
            If True and merge is True, check whether a point is shared with the node to remove in the children and remove this point if so. 
            
        Returns
        -------
        None.

        """
        
        if node2remove is None:
            node2remove = self.get_node(node_id)
        
        node_id2remove = node2remove.node_id
                
        if node_id2remove != node_id:
            print("Different node id asked ({0}) and selected to remove ({1})".format(node_id, node_id2remove))
        
        if node2remove.parent is not None:
            if merge:
                for child in node2remove.children:
                    node2remove.parent.add_child(child, offset=offset) #here is the conflct with parent method 
                    
            parent_id = node2remove.parent.node_id
            node2remove.parent.children.remove(node2remove)
            node2remove.parent.node_connections.remove([parent_id, node2remove.node_id])
        
            if update:
                root = self.get_root()
                root.update_ids(node_id2remove)
        
        node2remove.__del__()
        
        return
        
    
    def move_node(self, node2move, targetparent, move_subtree = False, offset = False, update = True):
        """
        
        Move a node either with its descendants or alone, and add it to a 
        the children list of a given target node.
        
        Parameters
        ----------
        node2move    : Tree
            The node that has to be moved.
            
        targetparent : Tree
            The happy parent node, to whom we add the node to move as child.
            
        move_subtree : bool, optional (Default : False)
            If True, all the descendant follow the node to move.
            If False, the node to move is bypassed
            
        offset      : bool, optional (Default : False)
            If True, and if merge is True, will translate the subtree to match the new parent's las point.
            
        update      : bool, optional (Default : True)    
            If True, update the branches ids and connections once the node removed.
            
        Returns
        -------
        None.
        
        """
        assert isinstance(targetparent,VascularTree), "Error : the specified parent is not a Tree object"
        assert isinstance(node2move,VascularTree), "Error : the node to move is not a Tree object"
        
        merge_descendent = not move_subtree
        
        node_tmp = copy.deepcopy(node2move)
        id_stored = node2move.node_id
        
        self.remove_node(id_stored,node2remove=node2move,merge=merge_descendent, update = False)
        
        node_tmp.node_id = id_stored
        if merge_descendent:
            node_tmp.children = []
            node_tmp.node_connections = []
        
        targetparent.add_child(node_tmp, offset = offset)
        
        if update:
            root = self.get_root()
            root.clean_ids()

        return
    
    
    def insert_node(self, new_node, target_children, update = False):
        """
        
        Insert a node at a given position. 
        
        Parameters
        ----------
        new_node        : Tree
            The node that has to be moved.
            
        target_children : list
            list of children node_ids.
            
        update          : bool, optional (Default : True)    
            If True, update the branches ids and connections once the node moved.

        Returns
        -------
        None
        """
        
        self.add_child(new_node, set_id = True, offset = True)
                
        ind_sel = []
                
        for i,child in enumerate(self.children):
            if child.node_id in target_children:
                ind_sel.append(i)
                
        cpt = 0
                
        for ind in ind_sel:
            node2move = self.children[ind-cpt]
            self.move_node(node2move, new_node, move_subtree = True, offset = True, update = update)
            #self.move_node(child, new_node, move_subtree = True, offset = True)
            cpt+=1
                
        return
         
    
    def remove_common_points(self):
        """
        Get rid of all the redundant points in the curves.
        
        """
        
        root = self.get_root()
        
        init_sampling = root.curve.data.shape[0]
        
        for b in root.descendants():
            
            if b.parent is not None:
                
                if (b.parent.curve.data[-1,:]==b.curve.data[0,:]).all():
                    b.curve.data = b.curve.data[1:,:]
                    
        root.resample_tree(init_sampling)
        
        return
        
    
    
    def resample_tree(self, new_sampling):
        """
        Resample the branches of the tree at a given number of points. 
        
        Parameters
        ----------
        new_length : int
            The new number of points.
            
        TODO : must deal with the empty nodes or nodes with only one point.
        """
        
        ind_first_point = 0
        root = self.get_root()
        
        for b in root.descendants():
            b.curve.resample_curve(new_sampling)
            b.curve.ind_first_point = ind_first_point
            ind_first_point += new_sampling    
            
        return
    
    
    def interpolate2target(self, target_data, t, graft = False):
        """
        Given a target branch, interpolate the current branch (seen as branch at t=0)
        and the target is the branch at t=1.
        
        Parameters
        ----------
        target_data : numpy ndarray
            The target branch.
            
        t           : float
            The time for the interpolation. 
            
        graft       : Bool, optional
            If True, replace the first point of self with the last of self.parent

        Returns
        -------
        None.
        
        
        """
            
        translation = t*(target_data[-1,:] - self.curve.data[-1,:])
        
        self.curve.interpolate2target(target_data, t)
        
        if graft and self.parent is not None:
            delta = self.parent.curve.data[-1,:] - self.curve.data[0,:]
            self.curve.data+=delta 
            translation+=delta

        descendants = self.descendants()
        descendants.pop(0)
        
        for node in descendants:
            node.curve.data+=translation
            
        return
    
    #%%TRIMMING
    def trim_tree_from_leaves(self, threshold):
        """
        Vascular Tree
        
        Trim the leaves of the tree smaller than a given length (threshold).
        If a single leaf remains, will be merged to its parent that will become
        the new leaf.
        
        Parameters
        ----------
        threshold : float
            The node that has to be merged with its parent.

        Returns
        -------
        None.
        """
        
        all_clear = False
        
        root = self.get_root()
        
        while all_clear is False:
            
            all_clear = True
            
            leaves_list = root.get_leaves([])
            for current_node in leaves_list:
                if current_node.parent is None:
                    return 
                else:    
                    parent = current_node.parent
                    if len(current_node.get_siblings())==1:
                        parent.merge_to_parent(current_node)
                    else:
                        siblings = current_node.get_siblings()
                        list_lengths = []
                        list_ind = []
                        cpt = 0
                        for sib in siblings:
                            
                            if sib.is_leaf():
                                list_ind.append(cpt)
                                list_lengths.append(sib.curve.get_length())
                                cpt+=1
                                
                        ind = list_ind[np.argmin(list_lengths)]
                        if list_lengths[ind]<=threshold:
                            root.remove_node(siblings[ind].node_id, merge = False)
                            all_clear = False        
                            for sib in siblings: #not sure if needed
                                if sib.is_leaf() and sib in leaves_list:
                                    leaves_list.remove(sib)
                                                
        root.update_connections(0)
        return
    
    
    #%%MERGING
    def merge_to_parent(self, child2merge):
        """
        Function to merge a node with its parent. 
        If this node has siblings, nothing will be done right now.
        
        Parameters
        ----------
        child2merge : Tree
            The node that has to be merged with its parent.

        Returns
        -------
        None.

        """
    
        if len(child2merge.get_siblings())>1:
            print("I do not know how to merge a branch with siblings.")
            return
        
        else:
            parent = child2merge.parent
            len_base = len(parent.curve.data)-1
            
            if len_base == 0:
                parent.curve.points_connections.pop(0)

            """if parent.label <0:
                parent.label = child2merge.label
            else:
                if child2merge.label != parent.label:
                    parent.label = -1"""
            parent.label = child2merge.label
            
            if (parent.curve.data[-1,:] == child2merge.curve.data[0,:]).all():
                parent.curve.data = np.concatenate((parent.curve.data, child2merge.curve.data[1:,:]))
                for i,point in enumerate(child2merge.curve.data[1:,:]):
                    parent.curve.points_connections.append([i+len_base,i+len_base+1])
            else:
                parent.curve.data = np.concatenate((parent.curve.data, child2merge.curve.data))
                for i,point in enumerate(child2merge.curve.data):
                    parent.curve.points_connections.append([i+len_base,i+len_base+1])
            self.remove_node(child2merge.node_id, child2merge, merge = True)
            
        return
    
    
    def merge_single_child(self):
        """
        Search through all the nodes if there are ones with a unique child. 
        If so, merge them.
        
        Returns
        -------
        None
        """
        
        leaves_list = self.get_leaves()
        
        for current_node in leaves_list:
            end = False
            while end is False:
                node_temp = current_node.parent
            
                if node_temp is None:
                    end = True
                else:    
                    if len(current_node.get_siblings())==1:
                        node_temp.merge_to_parent(current_node)
                current_node = node_temp
                #end of the while loop when we reach the root.
                    
        return
    
    
    #%% #### CONVERSIONS #####
    def transition_points_connections(self):
        """
        Returns the connection of the previous last point in the tree and the 
        first point of the current node.
        
        Returns
        -------
        transition_connection : list
            list of two integers.
        shared_point          : boolean
            Whether the parent and the current node share a point.
        """
        transition_connection = [0, 0]
        shared_point = False
        
        if self.curve.data.shape[0] != 0:
                        
            ind_previous, previous_point = self.get_previous_point(None)

            if ind_previous is not None:
                shared_point = (previous_point==self.curve.data[0,:]).all()
                
                if self.curve.data.shape[0]==1 and shared_point: #There is only one point in the node, and it is shared with the parent
                    return transition_connection, shared_point
                
                else:
                    if shared_point:
                        transition_connection = [ind_previous, self.curve.points_connections[0][1]]
                    else:
                        transition_connection = [ind_previous, self.curve.points_connections[0][0]]
                    
        return transition_connection, shared_point
    
    
    def Tree2Points(self, points = None, connections = [], labels = []):
        """
        Convert to a point representation (VTK like). To retrieve the points 
        tree representation, one may call: "nodes_branches_to_points".
        The dual function is "nodes_points_to_branches":
        
        Returns
        -------
        points      : numpy ndarray
            The points coordinates.
        connections : list
            List of lists of two integers : the connected points.
        labels      : list
            List of integers, the points labels.
        """
                 
        if points is None: #Then we initialize the data 
            self.update_connections(0)
            points = copy.deepcopy(self.curve.data)
            connections = copy.deepcopy(self.curve.points_connections)
            labels = [self.label for l in range(self.curve.data.shape[0])]

        transition_connection, shared_point = self.transition_points_connections()
        
        if transition_connection[0]==transition_connection[1] and self.parent is None:
            for child in self.children:
                points, connections, labels = child.Tree2Points(points, connections, labels)
            return points, connections, labels
            
        if shared_point:
            if transition_connection[0]!=transition_connection[1]:
                connections.append(transition_connection)
            points = np.concatenate((points, self.curve.data[1:,:]), axis = 0)
            connections +=  self.curve.points_connections[1:]
            labels      += [self.label for i in range(self.curve.data.shape[0])[1:]]
        else:
            connections.append(transition_connection)
            points = np.concatenate((points, self.curve.data), axis = 0)
            connections +=  self.curve.points_connections
            labels      += [self.label for i in range(self.curve.data.shape[0])]
    
        for child in self.children:
            points, connections, labels = child.Tree2Points(points, connections, labels)
    
        return points, connections, labels


    def nodes_points_to_branches(self):
        """
        Simple wrapper for explicit conversion. 
        Switch from a representation where the points of the tree are nodes
        to a reprensentation with branches as nodes.

        Returns
        -------
        None.

        """
        self.merge_single_child()
        return 
    
    
    def nodes_branches_to_points(self):
        """
        Swith the tree representation from nodes as branches to nodes as 
        points.

        Returns
        -------
        tree2 : Tree
            The point representation of the Tree.

        """
        points, connections, labels = self.Tree2Points()
        tree2 = recursive_tree_build_points(0,None,points, np.asarray(connections), labels)
        return tree2
        
        
    def nodes_to_curves_points(self, n_sampling):
        
        root = self.get_root()
        root.resample_tree(n_sampling)
        nodes_list = root.get_nodes_list([])
        
        dim = root.curve.data.shape[1]
        
        tensor = np.zeros((len(nodes_list),n_sampling,dim))
        
        for i,n in enumerate(nodes_list):
            
            node = root.get_node(n)
            tensor[i,:,:] = node.curve.data

        return tensor

    ##############################################################################
    ##################### FUNCTIONS COPIED FROM GEOMETREE ########################
    ##############################################################################
    
    def newick(self,intern=False):
        """
        Convert the tree to the newick representation. 
        In this syntax, the interior nodes have no name, and are defined by
        the set of leaves (splits) one can have by removing them.
        
        Parameters
        ----------
        intern : bool (Default : False)
            Whether the node is interior or not.
        bl     : bool (Default : False)
            Whether we want the length of the branches or not.

        Returns
        -------
        None.
        """
        if len(self.children)==0:
            return "{0}:{1}".format(self.node_id,self.curve.get_length())
        s="("+str(self.children[0].newick(intern))
        for i in range(1,len(self.children)):
            s+=","+str(self.children[i].newick(intern))
        s+=')'
        if self.parent is None:return s
        if intern:s+=str(self.node_id)
        s+=":{0}".format(self.curve.get_length())
        return s
    
    
    def full_newick(self,intern=False):
        """
        

        Parameters
        ----------
        intern : TYPE, optional
            DESCRIPTION. The default is False.
        bl : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        l = str(self.curve.get_length())
        s = '('+self.newick(intern=intern)+'0:'+l+')'

        return s
    
    
    def newick_labels(self):
        """
        Convert the tree to the newick representation. 
        In this syntax, the interior nodes have no name, and are defined by
        the set of leaves (splits) one can have by removing them.
        
        Parameters
        ----------
        intern : bool (Default : False)
            Whether the node is interior or not.
        bl     : bool (Default : False)
            Whether we want the length of the branches or not.

        Returns
        -------
        None.
        """
        if len(self.children)==0:
            return "{0}:{1}".format(self.label,self.curve.get_length())
        s="("+str(self.children[0].newick_labels())
        for i in range(1,len(self.children)):
            s+=","+str(self.children[i].newick_labels())
        s+=')'
        if self.parent is None:return s #'('+s+':'+str(self.curve.get_length())+')'
        s+=":{0}".format(self.curve.get_length())
        return s
    
    
#%%############################################################            
############ END OF CLASS, CONSTRUCTION FUNCTIONS #############
###############################################################  
def recursive_tree_build_points(ind,parent,V,F,L):
    """
    

    Parameters
    ----------
    ind : int
        The ID of the new node, here the index of the point in the point cloud.
    parent : Tree
        The parent of the current node.
    V : numpy array
        The points coordinates of shape (n, d).
    F : numpy array
        The connections between the points of shape (n-1, 2).
    L : list
        The points labels, length n.
    Returns
    -------
    tree : Tree instance
        A Tree containing all the branches computed from the point cloud and the connectivity between them.

    """
    
    t = VascularTree(data = V[ind,:], node_id = ind, ind_first_point = ind, label = L[ind], parent = parent)
    children_index = F[np.where(F[:,0]==ind)[0],:][:,1]
    for ci in children_index:
        child = recursive_tree_build_points(ci,t,V,F,L)
        t.add_child(child)
    return t   


def recursive_distance(node,d,distances):
    distances[node.node_id] = d
    if node.parent is not None:
        d +=  np.sqrt(((node.curve.data-node.parent.curve.data)**2).sum())
    for c in node.children:
        recursive_distance(c,d,distances)
        
        
def get_recursive_max_length(node,max_length,current_length=0,keep_size=False):

    if node.curve.data.size>0:
   
        new_length = current_length + node.curve.get_length()

        if new_length <= max_length:
            for c in node.children:
                get_recursive_max_length(c,max_length,new_length, keep_size = keep_size)

        else: # to remove must be smaller than the curve length
            to_remove = new_length - max_length
            node.shorten(to_remove, keep_size = keep_size)

    return node


def remove_leave_labels(tree, label, n_interp):
    
    ids = tree.get_nodes_list([])

    for _id in ids:
        if tree.get_node(_id).label == label:
            tree.remove_node(_id)
            break 

    tree.merge_single_child()
    tree.clean_ids(0)
    tree.resample_tree(n_interp)

    return tree


def remove_labels_list(tree, list_labels, n_interp):
    
    ids = tree.get_nodes_list([])

    print(ids)

    for label in list_labels:
        for _id in ids:
            if tree.get_node(_id).label == label:
                tree.remove_node(_id, merge=False)
                break 

    tree.merge_single_child()
    tree.clean_ids(0)
    tree.resample_tree(n_interp)

    return tree



def tree_from_vtk(path):
    
    points, connections, labels = import_vtk(path)
    tree = recursive_tree_build_points(0,None,points,connections,labels)
    tree.nodes_points_to_branches()
    tree.update_lengths()
    tree.clean_ids()
    tree.set_attribute('curve')
    
    return tree

#%%############################################################
###################### PLOT SECTION ###########################
###############################################################
    
dic_label2color = {

        -3 : "white",
        -2 : "grey",
        -1 : 'black',
        0 : 'red',
        1 : "darkorange",
        2 : "blue",
        3 : "yellow",
        4 : "purple",
        5 : "powderblue",
        6 : "palevioletred",
        7 : "chartreuse",
        8 : "green",
        9 : "darkred",
        10 : "cyan",
        11 : "lime",
        12 : "darkblue",
        13 : "magenta",
}


def plot_recursive(node,ax):
    if node.curve.data.size>0:
        
        if node.label>16:
            lab = int(str(node.label)[:-1])
        else:
            lab = node.label
                
        if len(node.curve.data.shape)==1:
            ax.plot([node.curve.data[0]],[node.curve.data[1]],[node.curve.data[2]],'.',color=dic_label2color[lab], linewidth=2.0)
        else:
            ax.plot(node.curve.data[:,0],node.curve.data[:,1],node.curve.data[:,2],color=dic_label2color[lab], linewidth=2.0)
        for c in node.children:
            plot_recursive(c,ax)
       
    return
         

def plot_recursive_max_length(node,ax,max_length,current_length=0,keep_size=False):
    if node.curve.data.size>0:
        
        if node.label>16:
            lab = int(str(node.label)[:-1])
        else:
            lab = node.label
                
        new_length = current_length + node.curve.get_length()

        if new_length <= max_length:
            if len(node.curve.data.shape)==1:
                ax.plot([node.curve.data[0]],[node.curve.data[1]],[node.curve.data[2]],'.',color=dic_label2color[lab], linewidth=2.0)
            else:
                ax.plot(node.curve.data[:,0],node.curve.data[:,1],node.curve.data[:,2],color=dic_label2color[lab], linewidth=2.0)
            
            for c in node.children:
                plot_recursive_max_length(c,ax,max_length,new_length)
        else: # to remove must be smaller than the curve length
            to_remove = new_length - max_length
            node.shorten(to_remove, keep_size = keep_size)

            if len(node.curve.data.shape)==1:
                ax.plot([node.curve.data[0]],[node.curve.data[1]],[node.curve.data[2]],'.',color=dic_label2color[lab], linewidth=2.0)
            else:
                ax.plot(node.curve.data[:,0],node.curve.data[:,1],node.curve.data[:,2],color=dic_label2color[lab], linewidth=2.0)

    return


def print_recursive(node):
    print(node.node_id, ' : ', node.curve.length)
    print("Leaves : ", node.get_leaves_labels(),"\n")
    for child in node.children:
        print_recursive(child)


def print_recursive_connections(node):
    print("Branches : ", node.node_connections)
    print("Points : ", node.curve.points_connections)
    for child in node.children:
        print_recursive_connections(child)
        

def recursive_plot_points_cmap(node,distances,ax):
    n = 256
    colors = plt.cm.jet(np.linspace(0,1,n))
    if node.parent is not None:
        ind_color = int((distances[node.node_id]/distances.max())*(n-1))
        ax.plot([node.curve.data[0],node.parent.curve.data[0]],
                [node.curve.data[1],node.parent.curve.data[1]],
                [node.curve.data[2],node.parent.curve.data[2]],color=colors[ind_color])
    for c in node.children:
        recursive_plot_points_cmap(c,distances,ax)
    return



    
