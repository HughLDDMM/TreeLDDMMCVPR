# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 08:36:19 2022
"""

from geodesic_tools      import follow_geodesic_interpolation
from VascularTree_class  import VascularTree, plot_recursive, plot_recursive_max_length
from matplotlib.widgets  import Slider
import matplotlib.pyplot as plt
import numpy             as np

import copy


def plot_tree_slide_depth(tree):
    """
    
    """

    #tree = tree.nodes_branches_to_points()
    max_length = tree.get_max_geodesic_length()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Trimmed tree')
    ax.view_init(elev=200, azim=-60)

    ax.margins(x=0)

    axcolor = 'lightgoldenrodyellow'
    axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

    s_time = Slider(axtime, 'Time', 0, max_length, valinit=0, valstep=1)    
    
    def update(val):
        depth = s_time.val
        ax.clear()
        ax.set_title('Tree evolution')
        tmp = copy.deepcopy(tree)
        plot_recursive_max_length(tmp, ax, depth)
        fig.canvas.draw_idle()

    s_time.on_changed(update)

    plt.show()

    return


def plot_geodesic(geod, tree, tree_target):
    
    fig = plt.figure()
    ax2 = fig.add_subplot(131, projection='3d')
    ax2.set_title('Source')
    ax2.view_init(elev=200, azim=-60)
    plot_recursive(tree, ax2) 
    
    ax = fig.add_subplot(132,projection='3d')
    ax.set_title('Interpolation')
    ax.view_init(elev=200, azim=-60)
    #plt.subplots_adjust(left=0.25, bottom=0.25)
    t0 = 0
    delta_t = 0.05
    ax.margins(x=0)
    
    new_tree = follow_geodesic_interpolation(geod, copy.deepcopy(tree), tree_target, target_time = t0)
    plot_recursive(new_tree, ax) 
    t0+=delta_t
    axcolor = 'lightgoldenrodyellow'
    axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    
    times = []
    trees = []
    time = t0
    while time < 1:
        times.append(time)
        new_tree = follow_geodesic_interpolation(geod, copy.deepcopy(tree), tree_target, target_time = time)
        trees.append(new_tree)
        time+=delta_t
            
    times = np.asarray(times)
    
    s_time = Slider(axtime, 'Time', 0, 1, valinit=t0, valstep=delta_t)    
    
    def update(val):
        t = s_time.val
        tmp = np.abs(times-t)
        ind = np.argmin(tmp)
        ax.clear()
        ax.set_title('Interpolation')
        #new_tree = follow_geodesic_interpolation(geod, copy.deepcopy(tree), tree_target, target_time = t0)
        plot_recursive(trees[ind],ax)
        fig.canvas.draw_idle() 
    
    ax3 = fig.add_subplot(133, projection='3d')
    plot_recursive(tree_target, ax3) 
    
    ax3.set_title('Target')
    ax3.view_init(elev=200, azim=-60)
    s_time.on_changed(update)

    plt.show()
    
    return trees
