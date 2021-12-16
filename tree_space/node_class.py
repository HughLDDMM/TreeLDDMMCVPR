# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 15:24:24 2021
"""

import copy
import numpy as np

import sys
sys.setrecursionlimit(100000)


class Node(object):
    
    
    def __init__(self, node = None, node_id = -1, attribute = 0, label = -1, parent = None, children = None):
        
        if node is not None:
            #Create a copy of a Tree instance
            assert isinstance(node,Node), "The initial node must be a proper class Node"
            self.__init__(node_id = copy.deepcopy(node.node_id),
                          attribute = copy.deepcopy(node.attribute),
                          label = copy.deepcopy(node.label),
                          parent = copy.deepcopy(node.parent), 
                          children = copy.deepcopy(node.children))  
            
        else: #Create a Tree instance from informations
            
            #Initialization   
            self.node_id = node_id       
            self.node_connections   = []
            self.children = []
            self.matrix_connections = None
            self.attribute = np.asarray([attribute])
            self.label = label
            
            #Link to the parent
            if parent is not None:
                assert isinstance(parent,Node), "Error : the specified parent is not a Node object"
            self.parent = parent
            
            #Set the node's depth
            if self.parent is not None:
                self.depth = self.parent.depth+1
            else:
                self.depth = 0
            
            #Add children
            if children is not None and children != []:
                for child in children:
                    assert isinstance(child,Node)
                    self.add_child(child)  

            for child in self.children:
                self.node_connections.append([self.node_id,child.node_id])
            

    def __del__(self):
        del self
        
        
    def del_all(self):
        del self.node_connections
        del self.matrix_connections
        for child in self.children:
            child.del_all()
        self.parent.del_all()    
    

    def set_id(self, new_id):
        """
        Set an ID if it does not exist in the tree
        """
        
        list_ids = self.get_nodes_list()
        if new_id in list_ids:
            print("This ID already exists, aborting.")
        else:
            self.node_id = new_id
        
        return


    def reset_ids(self, count):
        """
        Set an ID if it does not exist in the tree
        """
        if count == 0 and self.parent is not None:
            root = self.get_root()
            root.reset_ids(0)
        
        else:
            self.node_id = count 
            for b_con in self.node_connections:
                b_con[0] = count
            count+=1
            for i,child in enumerate(self.children):
                self.node_connections[i][1] = count
                count = child.reset_ids(count)
        
        return count


    def clean_ids(self, count = 0, force = False):
        """
        Check through all the tree to avoid node_id duplicated.
        If count is 0, and there are duplicates in the ids, or if force is True,
        will reset the nodes ids.
        """
        
        if not force: #Then we have to make a test first
            list_ids = self.get_nodes_list([])
            if len(list_ids)==len(np.unique(list_ids)):
                return
            
            else:
                if count == 0:
                    root = self.get_root()
                    root.node_id = 0
                    count += 1
                    for child in root.children:
                        child.clean_ids(count, force = True) # Set force to True because no need to check the id list anymore           
                else:
                    self.node_id = count
                    count+=1
                    for child in self.children:
                        child.clean_ids(count, force = True)
                    
                return
        else: #force is True and we reset the nodes ids
            if count == 0:
                root = self.get_root()
                root.node_id = 0
                count += 1
                for child in root.children:
                    child.clean_ids(count, force = force)                    
            else:
                self.node_id = count
                count+=1
                for child in self.children:
                    child.clean_ids(count, force = force)
                
            return
        

    def update_ids(self, removed_id):
        """
        When a node is removed, update the ids in consequence.
        """
        
        for connection in self.node_connections:
            
            for c in connection:
                if c > removed_id:
                    c-=1
                
        if self.node_id > removed_id:
            self.node_id-=1
        
        for b_con in self.node_connections:
            if b_con[0]>removed_id:
                b_con[0]-=1
            if b_con[1]>removed_id:
                b_con[1]-=1
                
        for child in self.children:
            child.update_ids(removed_id)
        
        return

    
    def __print__(self):
        print('Node : ', self.node_id)
        for child in self.children:
            child.__print__()      
        return
    
    
    #%% ACCESSORS
    def get_nodes_list(self, list_nodes_id):
        """
        List all the nodes id in the tree, starting at the root.

        Parameters
        ----------
        list_nodes_id : list
            The list of the nodes ids.

        Returns
        -------
        list_nodes_id : list
            The list of the nodes ids.

        """
        
        if list_nodes_id == []:
            root = self.get_root()
            list_nodes_id.append(root.node_id)
            for child in root.children:
                child.get_nodes_list(list_nodes_id)  
        else:
            list_nodes_id.append(self.node_id)
            for child in self.children:
                child.get_nodes_list(list_nodes_id)
                
        return list_nodes_id


    def get_position(self, target_id):

        nodes_list = self.get_root().get_nodes_list([])
        position = nodes_list.index(target_id)

        return position


    def get_interior_nodes_list(self, list_nodes_id = [], start = True):
        """
        List all the nodes id in the tree, starting at the root.

        Parameters
        ----------
        list_nodes_id : list
            The list of the nodes ids.

        Returns
        -------
        list_nodes_id : list
            The list of the nodes ids.

        """
        
        if start:
            root = self.get_root()
            tmp = []
            for child in root.children:
                tmp += child.get_interior_nodes_list(list_nodes_id = tmp, start = False)  
            return tmp
        
        else:
            if self.children == []:
                return []
            else:
                tmp = []
                for child in self.children:
                    tmp += child.get_interior_nodes_list(list_nodes_id = tmp, start = False)
                tmp = [self.node_id] + tmp
            return tmp
    

    def get_root(self):
        
        root = self
        if self.parent is not None and self.node_id!=0:
            root = self.parent.get_root()

        return root


    def get_node(self, node_id, start = 0):
        """
        WARNING : 
        The current implementation requires to search recursively through the tree.
        A faster method could be to build the connectivity matrix and automatically 
        get the node we are looking for.
        """
        target_node = None
        if self.node_id == node_id:
            return self
        else:
            for child in self.children:
                target_node = child.get_node(node_id) 
                if target_node is not None:
                    break

        return target_node


    def get_siblings(self):
        """
        
        """
        
        siblings = None
        if self.parent is not None:
            siblings = self.parent.children
        return siblings
    
    
    def is_leaf(self):
        """
        Simple wrapper to check whether the node is a leaf.

        Returns
        -------
        bool
            Whether the branch is a leaf or not.

        """
        return self.children==[]
    
        
    def get_leaves(self):
        """
        
        """
        
        if self.children == []:
            return [self]
        leaves_list = []
        for child in self.children:
            leaves_list += child.get_leaves()
        
        return leaves_list
    
    
    def get_leaves_IDs(self):
        """
        Returns list of leaves IDs under the current node.
        """
        
        if len(self.children)==0:return [self.node_id]
        l=[]
        for child in self.children:
            l+=child.get_leaves_IDs()
        return l     
    
    
    def get_leaves_labels(self):
        """
        Returns list of leaves IDs under the current node.
        """
        
        if len(self.children)==0:return [self.label]
        l=[]
        for child in self.children:
            l+=child.get_leaves_labels()
        return l
    
    
    def ancestors(self): 
        """
        Return ancestor nodes and self node in order root->bottom

        Returns
        -------
        list
            The list of node ids from the root to the starting node.
        """
        if self.parent==None: 
            return [self.node_id]
        else: 
            return self.parent.ancestors()+[self.node_id]
        
        
    def descendants(self):
        """
        Return descendant nodes and self node in order root->bottom

        Returns
        -------
        list
            The list of nodes from starting node to the leaves.
        """
        if self.children==[]: 
            return [self]
        else: 
            tmp = [self]
            for child in self.children:
                tmp+=child.descendants()
            return tmp
        
        
    def get_depth(self, depth = 0):
        """
        Node

        Parameters
        ----------
        depth : int, optional
            To initialize the depth count. The default is 0.

        Returns
        -------
        depth : int
            The depth of the node.

        """
        if self.parent is not None:
            depth+=1
            depth = self.parent.get_depth(depth = depth)
                
        return depth
        

    def get_max_depth(self):
        """
        
        """
        max_depth = 0
        root = self.get_root()
        leaves = root.get_leaves()

        for leaf in leaves:
            l_depth = leaf.get_depth() 
            if l_depth > max_depth:
                max_depth = l_depth

        return max_depth
    
    
    def connectivity_matrix(self, force = False):
        """
        
        Parameters
        ----------
        force : Boolean
            Whether one whant to force the nodes id reinitialization.

        Returns
        -------
        connections : numpy array
            (n_nodes x n_nodes) matrix with 1 at connected nodes.
        """
        self.clean_ids(count = 0, force = force)
        
        list_nodes = self.get_nodes_list([])
        
        connections = np.zeros(len(list_nodes))
        
        for i, node_id in enumerate(list_nodes):
            
            node = self.get_node(node_id)
            
            for child in node.children:
                ind_child_list = list_nodes.find(child.node_id)
                connections[i,ind_child_list] = 1
                connections[ind_child_list,i] = 1
                    
        return connections, list_nodes
    
    
    def lowest_common_ancestor(self, other_node):
        """
        
        Parameters
        ----------
        other_node : Tree
            The node for which 'self' is looking for their closest common ancestor.

        Returns
        -------
        LCA : Tree
            The closest common ancestor to self and other_node.
        """
        
        if self == other_node:
            return self
        
        LCA = self.get_root()
        
        self_depth = self.depth
        depth_other = other_node.depth
                
        if self_depth >= depth_other:
            parent      = self.parent
            other       = other_node
            #other_depth = depth_other
        else:
            parent      = other_node.parent
            other       = self
            #other_depth = self_depth
            
        list_ancestors = []
        found = False

        #First we climb the tree starting from the deepest node
        while parent is not None and found is False:
            if parent == other:
                LCA = parent
                found = True                
            else: #parent.depth <= other_depth:
                list_ancestors.append(parent)
                parent = parent.parent
                
                
        if found is True:
            return LCA 
        else: 
            #The proximal node was not an ancestor, hence we climb the tree 
            #from the highest node, and search in the first node's ancestors.
            parent = other.parent
            while parent is not None and found is False:
                if parent in list_ancestors:
                    found = True
                    LCA = parent
                else:
                    parent = parent.parent
        
        return LCA
    
    
    #%% ADD/REMOVE
    def add_child(self, node, set_id = False, prepend=False):
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
            the new parent's las point.
            
        Returns
        -------
        None.

        """
        
        assert isinstance(node,Node), "the parameter node is not an instance of Node class."
        
        if set_id:
            nodes_list = self.get_nodes_list([])
    
            if (node.node_id == -1 or node.node_id in nodes_list):
                print('Setting a new id')
                node.node_id = max(nodes_list)+1
        node.parent = self 
        if prepend:
            self.children.insert(0, node)
            self.node_connections.insert(0, [self.node_id,node.node_id])
        else:
            self.children.append(node)
            self.node_connections.append([self.node_id,node.node_id])
    
        return
    
    
    def add_node(self, node, parent_id = 0):
        assert isinstance(node,Node), "The new node must be a proper class Node"
        
        if self.data is None:
            print("Adding a node to an empty Node, initializing with this node")
            self.__init__(node = node)
        else:
            parent = self.get_node(parent_id)
            if parent is not None:
                parent.children.append(node)
                node.parent = parent
            else:
                print("Did not find the parent in the current node (root : {0})".format(self.node_id))
        return
    
        
    def delete_subtree(self, subtree_root_id, subtree_root = None):
        
        if subtree_root is None:
            subtree_root = self.get_node(subtree_root_id)
        
        for child in subtree_root.children:
            child.delete_subtree(child.node_id)
            
        if(subtree_root is None):
            print("Found nothing to remove")
            
        del subtree_root
        
        return
    
    
    
    def remove_node(self, node_id, node2remove = None, merge = True, update = True):
        """
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
            If True, and if merge is True, will translate the subtree to match the new parent's las point.
             
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
                    node2remove.parent.add_child(child)
                    
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
        move_subtree : bool
            If True, all the descendant follow the node to move.
            If False, the node to move is bypassed

        Returns
        -------
        None.
        
        """
        assert isinstance(targetparent,Node), "Error : the specified parent is not a Node instance"
        assert isinstance(node2move,Node), "Error : the node to move is not a Tree instance"
        
        merge_descendent = not move_subtree
        
        node_tmp = copy.deepcopy(node2move)
        id_stored = node2move.node_id
        
        self.remove_node(id_stored,node2remove=node2move,merge=merge_descendent, update = update)
        
        node_tmp.node_id = id_stored
        if merge_descendent:
            node_tmp.children = []
            node_tmp.node_connections = []
        
        targetparent.add_child(node_tmp, offset = offset)
        
        return
    
    
    def insert_node(self, new_node, target_children, update = False):
        """
        

        Parameters
        ----------
        new_node        : Tree
            The node that has to be moved.
        target_children : list
            list of children node_ids.

        Returns
        -------
        None
        """
        
        self.add_child(new_node, set_id = True)
                
        ind_sel = []
                
        for i,child in enumerate(self.children):
            if child.node_id in target_children:
                ind_sel.append(i)
                
        cpt = 0
                
        for ind in ind_sel:
            node2move = self.children[ind-cpt]
            self.move_node(node2move, new_node, move_subtree = True, update = update)
            cpt+=1
                
        return
    
    
    def find_smallest_split(self, list_leaves_ids):
        """
        Recursive function finding the most distal node leading to the set of 
        leaves list_leaves ids.
        
        Parameters
        ----------
        threshold : float
            The node that has to be merged with its parent.

        Returns
        -------
        next_split : Tree
            Returns a Tree if we found a node whose descendant leaves contains a split
        """
        
        current_leaves = self.get_leaves_labels()

        if set(list_leaves_ids).issubset(current_leaves):
                        
            for child in self.children:
                next_split = child.find_smallest_split(list_leaves_ids)
                if next_split is not None: #then we found a deeper node whose set of descendant leaves contains the list_leaves.
                    return next_split
                    break
                
            return self #then self is the deepest node containing the set of leaves we seek.
            
        return None
    
    
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

        Returns
        -------
        None.
        """
        if len(self.children)==0:
            return "{0}:{1}".format(self.node_id,self.get_length())
        s="("+str(self.children[0].newick(intern))
        for i in range(1,len(self.children)):
            s+=","+str(self.children[i].newick(intern))
        s+=')'
        if self.parent is None:return s
        if intern:s+=str(self.node_id)
        s+=":{0}".format(self.get_length())
        return s
    
    
    def full_newick(self,intern=False):
        """
        

        Parameters
        ----------
        intern : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        l = str(self.get_length())
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
            return "{0}:{1}".format(self.label,self.get_length())
        s="("+str(self.children[0].newick_labels())
        for i in range(1,len(self.children)):
            s+=","+str(self.children[i].newick_labels())
        s+=')'
        if self.parent is None:return s #'('+s+':'+str(self.get_length())+')'
        s+=":{0}".format(self.get_length())
        return s
    
