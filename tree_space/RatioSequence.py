# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:02:24 2021
"""
import copy
import numpy as np
from phylogenetic_distance import edge_average

"""
 A Ratio object represents some number of edges, which have been combined in the ratio sequence.
 They store list of edges from a first tree, and list of edges from a second tree. 
"""
class Ratio(object):
    
    def __init__(self, edges1 = None, edges2 = None, Tree1 = None, Tree2 = None):
        
        if edges1 is None:
            self.edges1 = []
            self.LenEdges1 = -1
            if Tree1 is not None :
                print("WARNING : Tree1 was provided but no corresponding edges. Init to emty ratio 1")
        else:
            self.edges1 = edges1
            self.LenEdges1 = len(edges1)
            
            if Tree1 is not None:
                length1 = self.get_length_1(Tree1)
                self.LenEdges1 = length1
                
        if edges2 is None:
            self.edges2 = []
            self.LenEdges2 = -1
            if Tree2 is not None :
                print("WARNING : Tree2 was provided but no corresponding edges. Init to emty ratio 2")
        else:
            self.edges2 = edges2
            self.LenEdges2 = -1
            
            if Tree2 is not None:
                length2 = self.get_length_2(Tree2)
                self.LenEdges2 = length2
                
                
    def set_length_1(self, l):
        self.LenEdges1=l
        return        
        
    def set_length_2(self, l):
        self.LenEdges2=l
        return 
    
    def get_length_1(self, Tree1):
        length1 = edge_average(Tree1, self.edges1)
        return length1
        
    
    def get_length_2(self, Tree2):
        length2 = edge_average(Tree2, self.edges2)
        return length2
    
    
    def update_lengths(self, Tree1, Tree2):
        length1 = edge_average(Tree1, self.edges1)
        self.set_length_1(length1)
        length2 = edge_average(Tree2, self.edges2)
        self.set_length_2(length2)
        return
    
    def get_ratio(self, Tree1, Tree2, update = False):
        
        if self.LenEdges1 == -1 or update:
            l1 = self.get_length_1(Tree1)
            self.set_length_1(l1)
        if self.LenEdges2 == -1 or update:
            l2 = self.get_length_2(Tree2)
            self.set_length_2(l2)
        
        r = self.LenEdges1 / self.LenEdges2
        
        return r
        
    
    def get_time(self, Tree1, Tree2, update = False):
        
        
        
        if self.LenEdges1 == -1 or update:
            l1 = self.get_length_1(Tree1)
            self.set_length_1(l1)
        if self.LenEdges2 == -1 or update:
            l2 = self.get_length_2(Tree2)
            self.set_length_2(l2)
        
        r = self.LenEdges1 / (self.LenEdges2+self.LenEdges1)
        
        return r
    
    
    def addAllEdges1(self, list_edges):
        self.edges1 += list_edges
        self.LenEdges1 = len(self.edges1)
        return
    
    
    def addAllEdges2(self, list_edges):
        self.edges2 += list_edges
        self.LenEdges2 = len(self.edges2)
        return
    
    def to_string(self):
        
        s = "{"
        for e in self.edges1:
            s+=str(e) + ','
            
        s = s[:-1] + "} / {"
        
        for e in self.edges2:
            s+=str(e) + ','
            
        s = s[:-1] + "}"
        
        return s

#END OF CLASS Ratio   
def combine_ratio(ratio1, ratio2):

    r = Ratio()
    if len(ratio1.edges1) == 0 and len(ratio2.edges1) == 0:
        r.set_length_1(np.sqrt( ratio1.LenEdges1**2 + ratio2.LenEdges1**2) )
    else:
        r.addAllEdges1(copy.deepcopy(ratio1.edges1))
        r.addAllEdges1(copy.deepcopy(ratio2.edges1))
        
    if len(ratio1.edges2) == 0 and len(ratio2.edges2) == 0:
        r.set_length_1(np.sqrt( ratio1.LenEdges2**2 + ratio2.LenEdges2**2) )
    else:
        r.addAllEdges2(copy.deepcopy(ratio1.edges2))
        r.addAllEdges2(copy.deepcopy(ratio2.edges2))
        
    return r



"""
This Class handles lists of ratios of splits, in order to sort the ratios and 
provide a geodesic path between trees.
"""
class RatioSequence(object):
    

    
    def __init__(self):
        
        self.ratio_list  = []
        self.combineCode = 0
        
        
    def size(self):
        return len(self.ratio_list)
    
    
    def setCombineCode(self, combineCode):
        assert len(combineCode)==self.size(), 'combineCode should have the size of ratio array -1 ({0}).'.format(self.size())
        self.combineCode = combineCode
        return
    
    
    def getNonDesRSWithMinDist(self, Tree1, Tree2):
        """
        Combines the ratios in rs so they are non-descending ie. so e1/f1 <= e2/f3 <= ... <= en/fn
	    Use the following algorithm: At the first pair of ratios which are descending, combine.  
	    Compare the combined ratio with the previous one, and combine if necessary, etc.
	 
	    If the ratio sequence we are running this on has size < 2, return that ratio sequence.

        Returns
        -------
        new_RS : RatioSequence
            DESCRIPTION.

        """
        if self.size() <2:
            return self
        
        combined_RS = copy.deepcopy(self)
        i           = 0 # index stepping through self
        combineCode = [0 for i in range(self.size()-1)]
        ccArray     = [2 for i in range(self.size()-1)]
        
        a           = 0 #index stepping throught ccArray
        
        while i < combined_RS.size()-1:
            if (combined_RS.ratio_list[i]).get_ratio(Tree1, Tree2) > (combined_RS.ratio_list[i+1]).get_ratio(Tree1, Tree2):
                #combine, remove the pair and update the Ratio Sequence
                combinedRatio = combine_ratio(combined_RS.ratio_list[i], combined_RS.ratio_list[i+1])    
                combined_RS.ratio_list.pop(i)
                combined_RS.ratio_list.pop(i)
                combined_RS.ratio_list.insert(i, combinedRatio)
                
                ccArray[a] = 1
                
                if i>0: #go to the last non-combined ratio
                    i-=1
                    while ccArray[a]==1: 
                        a-=1
                else: #we have to move a forward
                    while a < self.size()-1 and ccArray[a] != 2:
                        a +=1
                    
            else: #The ratio are non-descending, moving to the next pair
                ccArray[a] = 0
                i+=1
                while a < self.size()-1 and ccArray[a] != 2:
                    a +=1
        
        for k in range(self.size()-1):
            if ccArray[k] == 1:
                combineCode[k] = 1
                
        #combined_RS.setCombineCode(combineCode)

        #combined_RS._print(Tree1,Tree2)

        return combined_RS
    
    
    def get_distance(self,  Tree1, Tree2):
        
        d = 0
        for r in self.ratio_list:
            if r.LenEdges1== -1 or r.LenEdges2 == -1:
                print("In the ratio sequence distance computation, one of the length was not already computed, recomputing")
                r.update_lengths(Tree1, Tree2)
            d += (r.LenEdges1 + r.LenEdges2)**2
            
        return np.sqrt(d)
    
    def _print(self, T1 = None, T2 = None):
        
        for i,r in enumerate(self.ratio_list):
            if T1 is None or T2 is None:
                print('\n Ratio : {0}'.format(i))
            else: 
                print('\n Ratio : {0}, value = {1}, time = {2} '.format(i, r.get_ratio(T1, T2), r.get_time(T1,T2)))
            print(r.to_string())
            
        return
    
    
# END CLASS
def interleave(RS1, RS2, Tree1, Tree2):
    """
    Interleaves the ratio sequences rs1 and rs2 after combining them to get the 
    ascending ratio sequence with the min distance, to make a new ratio sequence.
    
    Parameters
    ----------
    RS1 : RatioSequence
        DESCRIPTION.
    RS2 : RatioSequence
        DESCRIPTION.
        
    Returns
    -------
    new_RS : RatioSequence
        DESCRIPTION.
    """
    
    combined1 = RS1.getNonDesRSWithMinDist(Tree1, Tree2)
    combined2 = RS2.getNonDesRSWithMinDist(Tree1, Tree2)
    
    interleavedRS = RatioSequence()
    
    ind1 = 0
    ind2 = 0
    
    while ind1 < combined1.size() and ind2 < combined2.size():
        if (combined1.ratio_list[ind1]).get_ratio(Tree1, Tree2) <= (combined2.ratio_list[ind2]).get_ratio(Tree1, Tree2):
            interleavedRS.ratio_list.append(combined1.ratio_list[ind1])
            ind1+=1
        else:
            interleavedRS.ratio_list.append(combined2.ratio_list[ind2])
            ind2+=1
            
    #Then we finish the lists:
    while ind1 < combined1.size():
        interleavedRS.ratio_list.append(combined1.ratio_list[ind1])
        ind1+=1
        
    while ind2 < combined2.size():
        interleavedRS.ratio_list.append(combined2.ratio_list[ind2])
        ind2+=1
        
        
    return interleavedRS




