# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Enrique_Escalante-Notario
# Instituto de Fisica, UNAM
# email: <enriquescalante@gmail.com>
# Distributed under terms of the GPLv3 license.
# data.py
# --------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D


head = ["Epoch", "Loss_train", "Accuracy_train", "Loss_validation", "Accuracy_validation"]


def plot_report(label, ext="pdf", epochs_initial=None, epochs_final=None):
    path_report = "./reports/log_loss_accuracy-"+label
    report_data = pd.read_csv(path_report, names=head)[epochs_initial:epochs_final]

    fig, (ax2,ax1) = plt.subplots(figsize=(20,15), nrows=2)

    report_data.plot('Epoch', 'Loss_train',linewidth=0.5,markevery=150, markersize=3, markeredgewidth=5,marker="v", ax=ax1)
    report_data.plot('Epoch', 'Loss_validation',linewidth=0.5,markevery=150, markersize=3, markeredgewidth=5,marker="^", ax=ax1)
    report_data.plot('Epoch', 'Accuracy_train',linewidth=0.5, markevery=150, markersize=3, markeredgewidth=5,marker="v", ax=ax2)
    report_data.plot('Epoch', 'Accuracy_validation',linewidth=0.5,markevery=150, markersize=3, markeredgewidth=5,marker="^", ax=ax2)
    #ax.set_title("Activation functions accuracy", loc="center",fontdict = {'fontsize':18, 'fontweight':'bold'})
    ax1.set_xlabel("Epochs",fontdict = {'fontsize':18})
    ax1.set_ylabel("Loss",fontdict = {'fontsize':20})
    ax2.set_xlabel("Epochs",fontdict = {'fontsize':18})
    ax2.set_ylabel("Accuracy",fontdict = {'fontsize':20})
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    ax1.legend(fontsize = 20,loc="upper right")
    ax2.legend(fontsize = 20, loc="lower right")
    fig.tight_layout()

    plt.savefig("./graphicsReport/acc_loss"+label+"."+ext, format=ext, transparent=True)
    
    plt.show()


def plot_success(label, ext="pdf"): 
    path = "./success/"+label
    success_data = pd.read_csv(path, header=None)

    names = ["feature_"+str(i) for i in range(1,len(success_data.columns))]
    names = ["total"]+names
    success_data.columns = names

    values = success_data.total.value_counts().sort_index()*100/len(success_data)

    fig, ax = plt.subplots(figsize=(15,27))
    values.plot(kind = "barh", width=0.9,ax=ax,color="Gray")
    ax.set_xlabel("Success percentage",fontdict = {'fontsize':40})
    ax.set_ylabel("Number of successful features",fontdict = {'fontsize':44})
    ax.xaxis.set_tick_params(labelsize=35,labelrotation=2)
    ax.yaxis.set_tick_params(labelsize=35)
    fig.tight_layout()
    plt.savefig('./graphicsSuccess/success'+label+"."+ext, format=ext, transparent=True)
    plt.show()


    

def plot_features_statistics(label, ext="pdf"):
    path = "./success/"+label
    success_data = pd.read_csv(path, header=None)
        
    names = ["feature_"+str(i) for i in range(1,len(success_data.columns))]
    names = ["total"]+names
    success_data.columns = names
    
    list_success = []
    
    for x in success_data.columns:
        if x != "total":
            l = success_data[x].value_counts().sort_index()*100/len(success_data)
            list_success.append(l[1])

        

    success_serie = pd.Series(list_success)

    fig, (ax,ax1) = plt.subplots(nrows=2,figsize=(20,15))
    
    success_serie.plot(kind = "bar", width=0.9, ax=ax)
    ax.set_xlabel("Feature",fontdict = {'fontsize':20})
    ax.set_ylabel("Success percentage",fontdict = {'fontsize':20})
    ax.xaxis.set_tick_params(labelsize=20,labelrotation=2)
    ax.yaxis.set_tick_params(labelsize=20)


    pd.Series(success_data.total.value_counts().sort_index()*100/len(success_data)).plot(kind='bar', width=0.9, ax=ax1)
    ax1.set_xlabel("Number of successful features",fontdict = {'fontsize':20})
    ax1.set_ylabel("Success percentage",fontdict = {'fontsize':20})
    ax1.xaxis.set_tick_params(labelsize=20,labelrotation=2)
    ax1.yaxis.set_tick_params(labelsize=20)
    
    plt.show()
    
    

def plot_2d_latent_space(latent_geo_z8, latent_geo_z12, ext="pdf", number_sample=1000):

    path_geo_z8 = "./latentSpaces/"+latent_geo_z8
    path_geo_z12 = "./latentSpaces/"+latent_geo_z12

    head = ['y1','y2','y3']

    dfGeo1 = pd.read_csv(path_geo_z8, sep=" ", names=head).sample(number_sample)
    dfGeo2 = pd.read_csv(path_geo_z12, sep=" ", names=head).sample(number_sample)



    fig, (ax1,ax2,ax3) = plt.subplots(figsize=(60,15),ncols=3)
    
    # firt plane
    dfGeo1.plot.scatter('y1','y2', label=r"Models from $\mathrm{\mathbb{Z}_8}$", ax=ax1)
    dfGeo2.plot.scatter('y1','y2', color="Red", label=r"Models from $\mathrm{\mathbb{Z}_{12}}$", ax=ax1)
    # second plane
    dfGeo1.plot.scatter('y1','y3', label=r"Models from $\mathrm{\mathbb{Z}_8}$", ax=ax2)
    dfGeo2.plot.scatter('y1','y3', color="Red", label=r"Models from $\mathrm{\mathbb{Z}_{12}}$", ax=ax2)
    # Thirth plane
    dfGeo1.plot.scatter('y1','y2', label=r"Models from $\mathrm{\mathbb{Z}_8}$", ax=ax3)
    dfGeo2.plot.scatter('y1','y2', color="Red", label=r"Models from $\mathrm{\mathbb{Z}_{12}}$", ax=ax3)


    ax1.set_xlabel("y1", fontdict = {'fontsize':36})
    ax1.set_ylabel("y2", fontdict = {'fontsize':40})
    ax2.set_xlabel("y1", fontdict = {'fontsize':36})
    ax2.set_ylabel("y3", fontdict = {'fontsize':40})
    ax3.set_xlabel("y2", fontdict = {'fontsize':36})
    ax3.set_ylabel("y3", fontdict = {'fontsize':40})
    ax1.xaxis.set_tick_params(labelsize=30)
    ax1.yaxis.set_tick_params(labelsize=30)
    ax2.xaxis.set_tick_params(labelsize=30)
    ax2.yaxis.set_tick_params(labelsize=30)
    ax3.xaxis.set_tick_params(labelsize=30)
    ax3.yaxis.set_tick_params(labelsize=30)
    ax1.legend(fontsize = 40,loc="upper right", markerscale=3., scatterpoints=1)
    ax2.legend(fontsize = 40, loc="upper right", markerscale=3., scatterpoints=1)
    ax3.legend(fontsize = 40, loc="upper right", markerscale=3., scatterpoints=1)
    fig.tight_layout()
        
    plt.savefig("./graphicsLatentSpace/"+latent_geo_z8[10:]+"."+ext, format='pdf', transparent=True)
    
    plt.show()

    
    