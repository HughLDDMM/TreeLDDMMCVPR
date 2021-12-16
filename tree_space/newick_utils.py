# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:40:41 2021
"""

from VascularTree_class import VascularTree, plot_recursive

import re


def parse_id_and_length(node_string):
    length = None
    if ':' in node_string:
        node_string,length = node_string.split(":",1)
    
    return node_string or None, length
    
    
def decompose(newick):
    """
    Decompose a newick string into a list of elements, either delimiters 
    (except ':') or strings as follow:
        'node_id:length'

    Parameters
    ----------
    newick : str
        A newick formated tree.

    Returns
    -------
    decomposition : list
        List of the different elements of the newick string.

    """
    split = [s for s in re.split('([,|(|)])', newick) if s !='']    
    return split
    

def depth_level(list_string):
    """
    

    Parameters
    ----------
    list_string : list
        Output of the funtion decompose(newick).

    Returns
    -------
    depth_list : list
        List of corresponding depth of the different elements of list_string.
        -1 is for the delimiters (except ':').

    """
    depth_list = []
    
    depth_level = -1 #the root is of depth 0
    
    for i,s in enumerate(list_string):
        if s == '(':
            depth_level+=1
            depth_list.append(-1)
        elif s==')':
            if i>1:
                if list_string[i-1]!=')':
                    depth_level-=1      
            depth_list.append(-1)
        elif s==',':
            depth_list.append(-1)
        else:
            depth_list.append(depth_level)
        
    
    return depth_list

    
def parse_newick(newick):
    """
    
    Takes a newick string as input, and converts to Tree object.
    For now, ALL branches must have a name/id and a length.
    
    TODO : adapt to unnamed branches and empty branches.
    
    Parameters
    ----------
    newick : string
        DESCRIPTION.

    Returns
    -------
    root : Tree
        The Tree object associated to the Newick string.
    
    """
    
    split = decompose(newick)[::-1]
    list_depth = depth_level(split)[::-1]
    
    
    root=VascularTree()
    root.length=0
    root.node_id = 0
    
    current_node = root
    new_node = root
    
    rooted = False
    
    for i,s in enumerate(split):
        if s not in ['(', ',', ')']:
            if list_depth[i]==0 and rooted is False:
                node_id, length =  parse_id_and_length(s)
                if length is not None:
                    root.length = length
                rooted = True
                current_node = root
            elif list_depth[i]==0:
                raise ValueError("There are several nodes at depth 0... Different trees?")
                
            else:
                node_id, length =  parse_id_and_length(s)
                new_node = VascularTree(node_id = node_id, parent = current_node)
                new_node.depth = list_depth[i]
                new_node.length = length
                current_node.add_child(new_node, prepend=True)
                
        elif s==',':
            pass
        elif s==')':
            current_node = new_node
        elif s=='(':
            current_node = current_node.parent
            
    return root


def create_data(node,length):
    
    #TODO
    
    return


def save_newick(path2save, newick_string):
    
    file1 = open(path2save,"w")       
    file1.write(newick_string) 
    file1.close() #to change file access modes 
    return


def plot_newick(newick_string, ax):
    
    root = parse_newick(newick_string)
        
    plot_recursive(root,ax)

    return








