from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, TransformedBbox,
                                                   BboxPatch, BboxConnector)
import pandas as pd
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import auc , roc_auc_score
import numpy as np, seaborn as sns

def my_mark_inset(parent_axes, inset_axes, loc1a=1, loc1b=1, loc2a=2, loc2b=2, **kwargs):
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)
    pp = BboxPatch(rect, fill=False, **kwargs)
    parent_axes.add_patch(pp)
    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)
    return pp, p1, p2
def subplotting(ax , store , x , y ,cond={}, **kwargs) :
    ax.plot(store[x],store[y],**kwargs)
    if "ylabel" in cond :
        ax.set_ylabel(cond["ylabel"], fontsize= 10)
    if "xlabel" in cond :
        ax.set_xlabel(cond["xlabel"], fontsize= 10)
    if "xlim" in cond :
        ax.set_xlim(cond["xlim"])
    if "ylim" in cond :
        ax.set_ylim(cond["ylim"])
    if "title" in cond :
        ax.set_title(cond["title"], fontsize= 15)
    ax.legend()
    return ax
## version3
def vis_onlysl(store:dict, path:str,title:str) :
    clear_output()
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(nrows=2, ncols=2)
    ax1 = fig.add_subplot(gs[0:1, 0])
    ax2 = fig.add_subplot(gs[0:1, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, 
            top=0.9, wspace=0.3, hspace=0.2)
    ax1.plot(store["epoch"],store["slloss"],label = "SLloss", color='c')
    ax1.set_xlabel('Epoch')
    select_sl= np.argmin(store["slloss"])
    ax1.vlines(store["epoch"][select_sl],
               np.min(store["slloss"]),np.max(store["slloss"]),
               label='Best', color='c')
    a =store["slloss"][-1]
    ax1.set_title(f'Loss : {a:.4f}')
    ax1.set_ylabel('Supervised', fontsize= 15)
    ax1.legend()
    ax2.plot(store["epoch"],store["auc"],label = "Train auc")
    ax2.plot(store["epoch"],store["teauc"],label = "Test auc")
    ax2.set_ylabel('AUC', fontsize= 15)
    ax2.set_xlabel('Epoch')
    select= np.argmax(store["teauc"])
    msg = f"test idx : {store['epoch'][select]}, maximum : {store['teauc'][select]*100:.2f}"
    ax2.set_title(msg)             
#     ax2.set_ylim(0.7,store['auc'][select]+0.1)
    ax2.legend()
    check_n = 4
    if len(store['epoch']) > check_n :
        axins = inset_axes(ax2, "100%", "100%", 
                           bbox_to_anchor=[0.36, .3, .5, .4],
                       bbox_transform=ax2.transAxes, borderpad=0)
        axins.plot(store['epoch'], store['auc'])
        axins.plot(store['epoch'], store['teauc'])
        maximum = np.max(store['auc'][-check_n:] + store['teauc'][-check_n:])
        minumum = np.min(store['auc'][-check_n:] + store['teauc'][-check_n:])
        xlims = (store['epoch'][-check_n],store['epoch'][-1])
        ylims = (minumum, maximum)
        axins.set(xlim=xlims, ylim=ylims)
        my_mark_inset(ax2, axins, loc1a=2, loc1b=3, loc2a=4, loc2b=4, fc="none", ec="0.5") # 
#     msg = f"Epoch : {epoch[-1]}, Loss : {loss[-1]:.3f}, Auc : {aucs[-1]:.3f}"
#     skplt.metrics.plot_ks_statistic(store["train_y"], store["train_prob"], 
#                                 ax = ax3 ,
#                                 title = "[Train] KS Static PLOT")
#     skplt.metrics.plot_ks_statistic(store["test_y"], store["test_prob"], 
#                                 ax = ax4 ,
#                                 title = "[Test] KS Static PLOT")
    sns.boxplot(x="t", y="prob", data=store["train_pd"] , ax = ax3)
    ax3.set_title("train" , fontsize= 15)
    sns.boxplot(x="t", y="prob", data=store["test_pd"] , ax = ax4)
    ax4.set_title("test" , fontsize= 15)
    plt.suptitle(title)
    plt.savefig(path)
    plt.show()
def pcolorplot(result:pd.DataFrame, save_path, title) :
    models = np.arange(result.shape[0]-2).tolist()
    models = models + ["pred","real"]
    indexs = np.arange(result.shape[1])
    fig, ax = plt.subplots( figsize = (12,8))
    plt.pcolor(s)
    plt.yticks(np.arange(0.5, len(models), 1),models)
    plt.title(title,fontsize=20)
    plt.savefig(save_path)
    plt.close()      