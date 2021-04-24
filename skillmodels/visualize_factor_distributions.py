
import pandas as pd 
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
import  seaborn as sns 
import itertools
from scipy import stats
import matplotlib.pyplot as plt
import scipy.stats as st
import re
import math





def plot_pairwise_factor_distributions(
    data_filter_state,
    period,
    combined_plots,
    add_3D_plots,
    n_points=50): 
    """Visualize pairwise_factor_distributions.
    Args:
        
        data_filter_state (pandas.DataFrame): Tidy DataFrame with filtered states.
            They are used to visualize pairwise_factor_distributions .
        period (int): The selected period of the filtered states that are plotted.
        combined_plots : decide whether to retrun a grid of plots or return a dict of individual plots
        add_3D_plots :decide whether to adda 3D plots in grid of plots or in the dict of individual plots
        n_points (int): Number of grid points per plot. For 3d plots this is per
            dimension.
    
    Returns:
        matplotlib.Figure: The grid plot or dict of seperate plots 
    """
    plot_dict={}
    if  combined_plots==True and add_3D_plots==True:
        plot_dict=plot_grid_2D_3D(data_filter_state, period,n_points)
    elif combined_plots==True and add_3D_plots==False:
        plot_dict=plot_grid_2D(data_filter_state, period,n_points)
    elif combined_plots==False and add_3D_plots==True:
        dict2=plot_2D_seperately(data_filter_state,period,n_points) 
        dict1=plot_3D_seperately(data_filter_state,period,n_points)
        plot_dict={**dict1, **dict2}
    else:
        plot_dict=plot_2D_seperately(data_filter_state,period,n_points)
    return plot_dict

def data_preparation(data_filter_state, period):
    
    data_period = data_filter_state.query(f"period == {period}")
    data_cleaned=data_period.drop(columns=['mixture', 'period', 'id'])
    names=data_cleaned.keys()
    factors=list(itertools.product(names, repeat=2))
    
    return factors, data_cleaned

def preparation_for_2D(ax, data_cleaned):
    upper_bound = math.ceil((data_cleaned.max()).max())
    lower_bound = math.floor((data_cleaned.min()).min())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set(xlim=(lower_bound, upper_bound), ylim=(lower_bound, upper_bound))      
    return ax 

def preparation_for_3D(ax, data_cleaned):
    upper_bound = math.ceil((data_cleaned.max()).max())
    lower_bound = math.floor((data_cleaned.min()).min())
    ax.set(xlim=(lower_bound, upper_bound), ylim=(lower_bound, upper_bound), zlim=(0, 1))
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_zlabel('KDE')
    ax.grid(False)
    ax.view_init(30, 35)
    
    return ax, lower_bound, upper_bound

def KDE_for_3D(data_cleaned,a, b, lower_bound, upper_bound):
    
    x=data_cleaned[a]
    y=data_cleaned[b]
    xx, yy = np.mgrid[lower_bound:upper_bound:50j, lower_bound:upper_bound:50j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    
    return xx, yy, f

def plot_grid_2D_3D(data_filter_state, period,n_points):
    
    factors, data_cleaned = data_preparation(data_filter_state, period)

    fig = plt.figure(figsize=(15,15))
    fig.suptitle(f'Grid of plots for Period {period}')
    size=int(np.sqrt(len(factors)))
    gs = fig.add_gridspec(size,size)
    for a,b in factors: 
        row=int(re.sub("[^0-9]","", a) ) 
        col=int(re.sub("[^0-9]","", b) )
        if row > col :
            ax = fig.add_subplot(gs[row-1, col-1])
            ax = preparation_for_2D(ax, data_cleaned)
            ax=sns.kdeplot(data_cleaned[a],data_cleaned[b],gridsize=n_points)
        
        elif row==col: 
            ax = fig.add_subplot(gs[row-1, col-1])
            ax = preparation_for_2D(ax, data_cleaned)
            ax=sns.kdeplot(data_cleaned[a],gridsize=n_points)
            ax.set(ylim=(0, 1.5))
        
        else:
            ax = fig.add_subplot(gs[row-1, col-1],projection='3d')
            ax, lower_bound, upper_bound = preparation_for_3D(ax, data_cleaned)
            xx, yy, f = KDE_for_3D(data_cleaned,a,b, lower_bound, upper_bound)
            surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, linewidth=0, antialiased=False, cmap='coolwarm', edgecolor='none')             
            fig.colorbar(surf, shrink=0.5, aspect=5)
    return fig

def plot_grid_2D(data_filter_state, period, n_points):

    factors, data_cleaned = data_preparation(data_filter_state, period)
    fig = plt.figure(figsize=(15,15))
    fig.suptitle(f'Grid of plots for Period {period}')
    size=int(np.sqrt(len(factors)))
    gs = fig.add_gridspec(size,size)
    for a,b in factors: 
        row=int(re.sub("[^0-9]","", a) ) 
        col=int(re.sub("[^0-9]","", b) )
        if row > col :
            ax = fig.add_subplot(gs[row-1, col-1])
            ax = preparation_for_2D(ax, data_cleaned)
            ax=sns.kdeplot(data_cleaned[a],data_cleaned[b],gridsize=n_points)
            
        elif row==col: 
            ax = fig.add_subplot(gs[row-1, col-1])
            ax = preparation_for_2D(ax, data_cleaned)
            ax=sns.kdeplot(data_cleaned[a],gridsize=n_points)
            ax.set(ylim=(0, 1.5))
    return fig
        
 

def plot_2D_seperately(data_filter_state,period,n_points): 
    figure_dict={}
    factors, data_cleaned = data_preparation(data_filter_state, period)
    for a,b in factors:
        row=int(re.sub("[^0-9]","", a) ) 
        col=int(re.sub("[^0-9]","", b) )
        if row == col :
            fig ,ax = plt.subplots()
            ax=sns.kdeplot(data_cleaned[a],gridsize=n_points)
            ax = preparation_for_2D(ax, data_cleaned)
            ax.set(ylim=(0, 1.2))
            fig.suptitle(f"{a}_2D_Period {period}")
            figure_dict[f'{a}_2D_Period {period}']=fig
        elif row < col: 
            fig ,ax = plt.subplots()
            ax=sns.kdeplot(data_cleaned[a],data_cleaned[b],girdsize=n_points)
            ax = preparation_for_2D(ax, data_cleaned)
            fig.suptitle(f"{a}_{b}_2D_Period {period}")
            figure_dict[f'{a}_{b}_2D_Period {period}']=fig
    return figure_dict
          
def plot_3D_seperately(data_filter_state,period,n_points):
    factors, data_cleaned = data_preparation(data_filter_state, period)
    fig = plt.figure(figsize=(15,15))
    size=int(np.sqrt(len(factors)))
    figure_dict={}
    
    for a,b in factors: 
        row=int(re.sub("[^0-9]","", a) ) 
        col=int(re.sub("[^0-9]","", b) )
        if row > col :
            fig = plt.figure()
            ax = plt.axes(projection='3d')  
            fig.suptitle(f'{a}_{b}_3D_Period {period}')
            ax, lower_bound, upper_bound = preparation_for_3D(ax, data_cleaned)
            xx, yy, f = KDE_for_3D(data_cleaned ,a,b,lower_bound, upper_bound)
            surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, linewidth=0, antialiased=False, cmap='coolwarm', edgecolor='none')             
            fig.colorbar(surf, shrink=0.5, aspect=5) 
            figure_dict[f'{a}_{b}_3D_Period {period}']=fig

    return figure_dict
    
    

     
        

