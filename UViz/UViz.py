
"""
Visualize UMAP results

Author: Caleb Hallinan

License: MIT
"""


##################################################################################################

from numpy.random import random_sample
import pandas as pd
import os
import numpy as np
import scipy.stats as stats
import time
# from sklearn.preprocessing import MinMaxScaler
import math
from scipy.spatial import distance
import umap
from sklearn.cluster import KMeans
from scipy.stats import zscore
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from scipy.stats import zscore
from sklearn.metrics.cluster import rand_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering

##################################################################################################


class UViz:
    def __init__(self, min_dist: int = 0.15, n_neighbors: int = 15, random_state: int = 0, standardize: str=False):
        self.min_dist = min_dist
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.standardize = standardize


    # UMAP function
    def UMAPing(self, data, y, features):
        """
        UMAP results from U-Het

        Parameters
        ----------
        data : pd.DataFrame of shape (n_samples, n_features)
            The input samples

        y : array-like of shape (n_samples,)
            The target values

        features : list
            List of features to use for UMAP

        min_dist : int (default=0)
            Minimum distance set for UMAP

        n_neighbors : int (default=15)
            Number of nearest neighbors for UMAP

        random_state : int (default=0)
            Random State

        standardize : bool (default=True)
            Standardize data before UMAP

        Returns
        -------
        matplotlib plot of UMAP-ed data
        """

        # standardize if needed
        if self.standardize:
            data = zscore(data)
        
        # check if dataframe, used to subset features later
        # if not isinstance(data, pd.DataFrame):
        #     raise Exception("Input data as pd.Dataframe with features as columns.")

        # make umap and umap data
        reducer_data = umap.UMAP(min_dist=self.min_dist, random_state=self.random_state, n_neighbors=self.n_neighbors)
        umap_data = reducer_data.fit_transform(data[features])

        # plot figure
        plt.figure(figsize=(8,6))
        sns.scatterplot(umap_data[:,0],umap_data[:,1], hue=y, palette='tab10')
        plt.xlabel('UMAP 1', fontname="Times New Roman Bold")
        plt.ylabel('UMAP 2', fontname="Times New Roman Bold")
        plt.title('UMAP Results', fontsize=16, fontname="Times New Roman Bold")
        plt.legend(title="Class")
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        sns.despine()


# TODO:
# 1. add ordering aspect to classes
# put circles further away from clusters


    # UMAP function
    def CLUSTERing(self, data, y, features, n_clusters=None, cluster_type="kmeans",reindex_order=None):
        """
        Clustering results from UMAP, will perform Silhoette Analysis

        Parameters
        ----------
        data : pd.DataFrame of shape (n_samples, n_features)
            The input samples

        y : array-like of shape (n_samples,)
            The target values

        features : list
            List of features to use for UMAP

        cluster_type : str (default=kmeans)
            Type of clustering algorithm to use for clustering data

        n_clusters : int (default=None)
            Pre-set number of clusters used, or leave as None to perform Silhoutte Analysis


        Returns
        -------
        matplotlib plot of UMAP-ed and Clustered data
        """

        # standardize if needed
        if self.standardize:
            data = zscore(data)
        
        # check if dataframe, used to subset features later
        # if not isinstance(data, pd.DataFrame):
        #     raise Exception("Input data as pd.Dataframe with features as columns.")

        # make umap and umap data
        reducer_data = umap.UMAP(min_dist=self.min_dist, random_state=self.random_state, n_neighbors=self.n_neighbors)
        umap_data = reducer_data.fit_transform(data[features])

        # make column names
        df_umap = pd.DataFrame(umap_data)
        df_umap.columns = ['umap1', 'umap2']

        # Perform Silhoette Analysis
        silhouette_avg_n_clusters = []

        if n_clusters == None:
            for i in range(2, 10):
                # do clustering first
                if cluster_type == 'kmeans':
                    clusterer = KMeans(n_clusters=int(i), random_state=self.random_state).fit(umap_data)
                if cluster_type == 'sc':
                    clusterer = SpectralClustering(n_clusters=int(i), random_state=self.random_state).fit(umap_data)
                if cluster_type == 'agg':
                    clusterer = AgglomerativeClustering(n_clusters=int(i)).fit(umap_data)

                # find silhoette score
                silhouette_avg = silhouette_score(umap_data, clusterer.labels_)

                print("For n_clusters =", i, "The average silhouette_score is :", silhouette_avg)

                # append and use highest later
                silhouette_avg_n_clusters.append(silhouette_avg)

            # use highest silhoette score for clusters
            tmp = max(silhouette_avg_n_clusters)
            n_clusters = silhouette_avg_n_clusters.index(tmp) + 2  # add 2 here because start clustering at 2
            print('Using ' + str(n_clusters) + ' to cluster data..')
            print("Silhoette Score " + str(tmp))

            if cluster_type == 'kmeans':
                kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state).fit(umap_data)
                df_umap['cluster'] = kmeans.labels_ + 1
            if cluster_type == 'sc':
                kmeans = SpectralClustering(n_clusters=n_clusters, random_state=self.random_state).fit(umap_data)
                df_umap['cluster'] = kmeans.labels_ + 1
            if cluster_type == 'agg':
                clusterer = AgglomerativeClustering(n_clusters=int(i)).fit(umap_data)
                df_umap['cluster'] = kmeans.labels_ + 1

        # or do own cluster number
        else:
            if cluster_type == 'kmeans':
                kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state).fit(umap_data)
                df_umap['cluster'] = kmeans.labels_ + 1
            if cluster_type == 'sc':
                kmeans = SpectralClustering(n_clusters=n_clusters, random_state=self.random_state).fit(umap_data)
                df_umap['cluster'] = kmeans.labels_ + 1
            tmp = 0

        # add class to dataframe
        df_umap['class'] = y

        ### Proportion ###
        for i in range(1,len(np.unique(df_umap['cluster']))+1):
            tmp_cluster = df_umap[df_umap['cluster'] == i]
            vc = pd.DataFrame(tmp_cluster['class'].value_counts())
            if i == 1:
                vc.columns = [i]
                prop = vc.copy()
            else:
                vc.columns = [i]
                prop = pd.concat([prop, vc],axis=1)
        prop = prop.fillna(0)
        if reindex_order != None:
            prop = prop.reindex(reindex_order)
        # stacked_data = prop.T.apply(lambda x: x*100/sum(x), axis=1)
        # stacked_data.plot(kind="bar", stacked=True)
        # plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0., fontsize=12,title='Class')

        # ### Proportion ###
        # for i in range(1,len(np.unique(df_umap['class']))+1):
        #     tmp_class = np.unique(df_umap['class'])[i-1]
        #     tmp_cluster = df_umap[df_umap['class'] == tmp_class]
        #     vc = pd.DataFrame(tmp_cluster['cluster'].value_counts())
        #     if i == 1:
        #         vc.columns = [tmp_class]
        #         prop1 = vc.copy()
        #     else:
        #         vc.columns = [tmp_class]
        #         prop1 = pd.concat([prop1, vc],axis=1)
        # prop1 = prop1.fillna(0)
        # # order prop if wanted
        # prop1 = prop1.sort_index()

        # plt.figure(figsize=(8,6))
        # stacked_data = prop1.T.apply(lambda x: x*100/sum(x), axis=1)
        # stacked_data.plot(kind="bar", stacked=True)
        # plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0., fontsize=12,title='Cluster')
        # plt.xlabel('Class')
        # plt.ylabel('Percent')
        # plt.title('Distribution of Clusters in each Class')


        # plot UMAP
        plt.figure(figsize=(8,6))
        sns.scatterplot(umap_data[:,0],umap_data[:,1], hue=y, palette='tab10', hue_order=reindex_order)
        plt.xlabel('UMAP 1', fontname="Times New Roman Bold")
        plt.ylabel('UMAP 2', fontname="Times New Roman Bold")
        plt.title('UMAP Results', fontsize=16, fontname="Times New Roman Bold")

        plt.legend(title="Class")
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        sns.despine()


        fig, ax1 = plt.subplots(figsize=(8,6))
        sc = sns.scatterplot(df_umap['umap1'], df_umap['umap2'], hue=df_umap['cluster'], palette='tab10')
        plt.xlabel('UMAP 1', fontname="Times New Roman Bold")
        plt.ylabel('UMAP 2', fontname="Times New Roman Bold")
        plt.title('Clustering Results', fontsize=16, fontname="Times New Roman Bold")
        # plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        sns.despine()

        h1, l1 = ax1.get_legend_handles_labels()
        # ax2.get_legend_handles_labels()

        # plotting pie charts one at a time
        for i in range(1, len(np.unique(df_umap['cluster']))+1):
            # get labels
            labels = list(prop[i].index)
            # get size of each pie
            sizes = list(prop[i])
            # get tmp data to calculate midway points to plot charts
            tmp_data = df_umap[df_umap['cluster'] == i]
            tmp_x1 = float(tmp_data[tmp_data['umap1'] == tmp_data['umap1'].max()]['umap1'])
            tmp_x2 = float(tmp_data[tmp_data['umap1'] == tmp_data['umap1'].min()]['umap1'])
            tmp_y1 = float(tmp_data[tmp_data['umap2'] == tmp_data['umap2'].max()]['umap2'])
            tmp_y2 = float(tmp_data[tmp_data['umap2'] == tmp_data['umap2'].min()]['umap2'])
            midpoint_x = (tmp_x1 + tmp_x2)/2
            midpoint_y = (tmp_y1 + tmp_y2)/2
            tmp_x, tmp_y = midpoint_x, midpoint_y
            xt = (tmp_x - ax1.get_xticks().min()) / (ax1.get_xticks().max() - ax1.get_xticks().min())
            yt = (tmp_y - ax1.get_yticks().min()) / (ax1.get_yticks().max() - ax1.get_yticks().min())
            # plot
            l, b, h, w = .1, .1, .1, .1
            ax2 = fig.add_axes([xt, yt, w, h])
            wedges, texts = plt.pie(sizes, shadow = True,startangle = 90,textprops={'fontsize': 10})
            plt.text(-.5,.0, "C"+str(i), fontsize=10, weight="bold")
            ax2.get_legend_handles_labels()
            ax1.legend(h1 + wedges, l1 + labels, title="Cluster", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)






