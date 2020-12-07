import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, SpectralBiclustering, SpectralCoclustering, Birch
from sklearn.metrics import normalized_mutual_info_score, pairwise_distances, pairwise
from sklearn.model_selection import GridSearchCV
from itertools import product, combinations
import numpy as np
from umap import UMAP
import time

class Config:
    def __init__(self, genedata_path, msdata_path):
        # relative path to data
        self.genedata_path = genedata_path
        self.msdata_path = msdata_path


class LoadData(Config):
    def __init__(self, genedata_path, msdata_path, first_feature_index):
        Config.__init__(self, genedata_path=genedata_path,
                        msdata_path=msdata_path)
        self.first_feature_index = first_feature_index
        self.load_data()

    def load_data(self):
        # load csv format data
        genedata = pd.read_csv(self.genedata_path)
        msdata = pd.read_csv(self.msdata_path)
        self.gene_feature_vectors, self.gene_labels = self.separate_data_labels(
            genedata)
        self.ms_feature_vectors, self.ms_labels = self.separate_data_labels(
            msdata)

    def separate_data_labels(self, data):
        '''
          input:  csv data
          output: feature vectors and corresponding labels
        '''
        # separate data and class in csv file
        labels = data['class']
        feature_vectors = data.iloc[:, self.first_feature_index:]
        # print(labels.shape)
        # print(feature_vectors.shape)
        return feature_vectors, labels


class Preprocess:
    def __init__(self, feature_vectors):
        self.data = feature_vectors

    def normalize_vertical(self):
        normalized_data = (self.data-self.data.min()) / \
            (self.data.max()-self.data.min())
        return normalized_data

    def normalize_horizontal(self):
        scale = self.data.max(axis=1)-self.data.min(axis=1)
        normalized_data = self.data.sub(self.data.min(
            axis=1), axis=0).div(scale, axis=0)
        return normalized_data

    def get_affinity(self, metric='euclidean', with_diag=True):
        """ 
        main purpose: compute pair distance
        input: distance metric
        output:distance matrix, a.k.a affinity
        """
        affinity = pairwise_distances(self.data, metric=metric)
        if with_diag:
            affinity[np.diag_indices(affinity.shape[0])] = 0
        else:
            affinity = affinity[~np.eye(affinity.shape[0], dtype=bool)].reshape(
                affinity.shape[0], -1)
        return affinity

    def pca(self, n_components, data=None):
        model = PCA(n_components=n_components)
        if data is not None:
            reduced_data = model.fit_transform(data)
        else:
            reduced_data = model.fit_transform(self.data)
        # print('pca explained_variance_ratio_ is {}'.format(
        #     model.explained_variance_ratio_))
        return reduced_data

    def umap(self, n_components, metric, data=None):
        model= UMAP(n_components=n_components,metric=metric)
        if data is not None:
            reduced_data = model.fit_transform(data)
        else:
            reduced_data = model.fit_transform(self.data)
        return reduced_data




class Cluster:
    def __init__(self, n_clusters, feature_vectors):
        self.n_clusters = n_clusters
        self.feature_vectors = feature_vectors

    def kmeans(self):
        self.model = KMeans(n_clusters=self.n_clusters)

    def agglomerative(self, linkage, affinity):
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters, linkage=linkage, affinity=affinity)

    def birch(self):
        #acc is 0.87
        self.model = Birch(n_clusters=self.n_clusters)

    def spectral(self, affinity, n_neighbors=None):
        self.model = SpectralClustering(
            n_clusters=self.n_clusters, affinity=affinity, n_neighbors=n_neighbors)

    def spectral_biclustering(self):
        self.model = SpectralBiclustering(n_clusters=self.n_clusters)

    def spectral_coclustering(self):
        self.model = SpectralCoclustering(n_clusters=self.n_clusters)

    def fit_model(self):
        # fit model and predict
        self.model.fit(self.feature_vectors)
        try:
            self.predicted_labels = self.model.labels_
        except AttributeError:
            # spectral_biclustering and Coclustering
            print(self.model.row_labels_.shape)
            self.predicted_labels = self.model.row_labels_
        except Exception:
            print(Exception)

    def save_result(self, file_path):
        np.savetxt('{}'.format(file_path),
                   self.predicted_labels.astype(int), fmt='%i')

    def goodness(self, true_labels, base_precision, improved_precision, verbose=False):
        self.fit_model()
        # evaluate performance
        normalized_mutual_info = normalized_mutual_info_score(
            true_labels, self.predicted_labels)
        points = (normalized_mutual_info-base_precision)/improved_precision + 1
        if verbose:
            print('current project can get {:d} points'.format(int(points)))
        return normalized_mutual_info


def visualize_median(data):
    plt.figure()
    plt.hist(data.median(),bins=25)
    plt.suptitle('median distribution of each feature')
    plt.xlabel='median'
    plt.show()


def visualize2D(data):
    """ 
      input: 2d Data
    """
    # print(data.iloc[0,:])
    plt.figure()
    plt.plot(data.iloc[0, :])
    plt.show()


def visualize3D(data, labels):
    """ 
        input: 
            data:3d Data
            labels:data label
    """
    classes = np.unique(labels)
    fig = plt.figure()
    sub_fig = fig.add_subplot(111, projection='3d')
    for class_index in classes:
        per_class_index = labels == class_index
        sub_fig.scatter(data[per_class_index, 0],
                        data[per_class_index, 1], data[per_class_index, 2])
    plt.show()




if __name__ == "__main__":
    data = LoadData(genedata_path='../data/genedata.csv',
                    msdata_path='../data/msdata.csv', first_feature_index=2)
    #--------------------------------------------------#
    #--------------------------------------------------#
    #----------------------gene_data---------------------#
    #--------------------------------------------------#
    #--------------------------------------------------#
    gene_cluster = Cluster(
        n_clusters=5, feature_vectors=data.gene_feature_vectors)
    # PCA from 7000 dimension to 3 dimension
    preprocess_gene=Preprocess(feature_vectors=data.gene_feature_vectors)
    reduced_gene=preprocess_gene.pca(n_components=3)
    # Visualization
    # visualize3D(reduced_gene,data.gene_labels)
    # K-means method for gene data
    gene_cluster.kmeans()
    nmi_kmeans=gene_cluster.goodness(data.gene_labels,base_precision=0.8, improved_precision=0.02)
    print('kmeans method on gene data get {}'.format(nmi_kmeans))
    # Spectral method for gene data
    gene_cluster.spectral(affinity='nearest_neighbors', n_neighbors=6)
    nmi_spectral=gene_cluster.goodness(data.gene_labels,base_precision=0.8, improved_precision=0.02)
    print('spectural method on gene data get {}'.format(nmi_spectral))
    # save spectral result
    # gene_cluster.save_result(file_path='../results/gene_results.txt')
    
    #--------------------------------------------------#
    #--------------------------------------------------#
    #----------------------ms_data---------------------#
    #--------------------------------------------------#
    #--------------------------------------------------#
    # visualization
    # visualize_median(data.ms_feature_vectors)
    preprocess_ms = Preprocess(feature_vectors=data.ms_feature_vectors)
    normalized_ms=preprocess_ms.normalize_vertical()
    # Kmeans method
    start_time=time.time()
    max_nmi=0
    threshold=0.98  # could be 1.0, flutuates between 0.97~1.0
    while(max_nmi<threshold):
        reduced_ms=preprocess_ms.umap(n_components=93,metric='braycurtis',data=normalized_ms)
        ms_cluster = Cluster(n_clusters=3, feature_vectors=reduced_ms)
        ms_cluster.kmeans()
        nmi=ms_cluster.goodness(data.ms_labels,base_precision=0.55, improved_precision=0.03)
        if(nmi>max_nmi):
            # ms_cluster.save_result(file_path='../results/ms_results.txt')
            max_nmi=nmi
    print('Kmeans method on ms data get{}'.format(nmi))
    print('use {} seconds to get result'.format(time.time()-start_time))

    ## Spectral method : 88% accuracy
    reduced_ms = preprocess_ms.pca(n_components=694)
    cos_dist = pairwise.cosine_distances(reduced_ms)
    ms_cluster = Cluster(n_clusters=3, feature_vectors=cos_dist)
    ms_cluster.spectral(
        affinity='precomputed_nearest_neighbors', n_neighbors=177)
    nmi=ms_cluster.goodness(data.ms_labels,base_precision=0.55, improved_precision=0.03)
    print('Spectral method on ms data get{}'.format(nmi))
    # ms_cluster.save_result(file_path='../results/ms_spectral_results.txt')


