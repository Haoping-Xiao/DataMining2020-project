import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, SpectralBiclustering, SpectralCoclustering, Birch
from sklearn.metrics import normalized_mutual_info_score, pairwise_distances
from sklearn.model_selection import GridSearchCV
from sklearn_extra.cluster import KMedoids
from itertools import product, combinations
import numpy as np


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

    def normalize(self):
        self.data = (self.data-self.data.mean())/self.data.std()
        return self.data

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

    def pca(self, n_components):
        model = PCA(n_components=n_components)
        reduced_data = model.fit_transform(self.data)
        # print('pca explained_variance_ratio_ is {}'.format(
        #     model.explained_variance_ratio_))
        return model.explained_variance_ratio_, reduced_data


def visualize2D(data):
    """ 
      input: 2d Data
    """
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1])
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


class Comparison:
    def __init__(self, feature_vectors, labels, metric, x_range):
        self.feature_vectors = feature_vectors
        self.labels = labels
        self.metric = metric
        self.x_range = x_range
        self.classes = np.unique(labels)

    def hist(self, data, title, row, col):
        fig, axes = plt.subplots(row, col)
        axes = axes.ravel()
        for index in range(len(data)):
            axes[index].set_title(index+1)
            axes[index].hist(data[index], range=self.x_range, bins=25)
        fig.suptitle(title)
        plt.show()

    # def get_mean_nearest_dist(self, pair_distance, k_nearest):
    #     sort_pair_distance = np.sort(pair_distance, axis=1)
    #     res = np.mean(sort_pair_distance[:, :k_nearest-1], axis=1)
    #     return res

    def get_affinity(self, data, with_diag=True):
        """ 
        main purpose: compute pair distance
        input: distance metric
        output:distance matrix, a.k.a affinity
        """
        affinity = pairwise_distances(data, metric=self.metric)
        if with_diag:
            affinity[np.diag_indices(affinity.shape[0])] = 0
        else:
            affinity = affinity[~np.eye(affinity.shape[0], dtype=bool)].reshape(
                affinity.shape[0], -1)
        return affinity

    def compare_pair_dist(self, isShow):
        dist_list = []
        plot_list = []
        # 1. all pair distance
        # preprocessed_data=Preprocess(feature_vectors=data)
        pair_distance = self.get_affinity(
            data=self.feature_vectors, with_diag=False)
        dist_list.append(pair_distance)
        plot_list.append(pair_distance.flatten())
        # 2. pair distance within each calss
        classes = np.unique(self.labels)
        for class_index in classes:
            # preprocessed_per_class_data=Preprocess(feature_vectors=data[labels==class_index])
            pair_distance_per_class = self.get_affinity(
                data=self.feature_vectors[self.labels == class_index], with_diag=False)
            dist_list.append(pair_distance_per_class)
            plot_list.append(pair_distance_per_class.flatten())
        if isShow:
            self.hist(data=plot_list, title='compare_pair_dist',
                      row=2, col=2)
        return dist_list

    def compare_nearest_dist(self):
        # dist_list = self.compare_pair_dist(isShow=False)
        top_k = 10
        plot_list = []
        pair_distance = self.get_affinity(
            data=self.feature_vectors, with_diag=False)

        sorted_distance = np.sort(pair_distance, axis=1)
        mean_nearest_dist = np.mean(sorted_distance[:, :top_k-1], axis=1)
        for class_index in self.classes:
            # print(mean_nearest_dist[self.labels==class_index].shape)
            plot_list.append(mean_nearest_dist[self.labels == class_index])
        self.hist(data=plot_list, title='compare_nearest_dist',
                  row=2, col=2)
        # return nearest_dist

    def compare_between_class_dist(self):
        classes = np.unique(self.labels)
        plot_list = []
        for exclude_class_index in classes:
            # preprocessed_data=Preprocess(feature_vectors=data[self.labels!=exclude_class_index])
            pair_distance_between_class = self.get_affinity(
                data=self.feature_vectors[self.labels != exclude_class_index], with_diag=False)
            # print(pair_distance_between_class)
            plot_list.append(pair_distance_between_class.flatten())
        self.hist(plot_list, title='compare_between_class_dist',
                  row=2, col=2)

    def compare_nearest_dist_other_class(self):
        classes = np.unique(self.labels)
        plot_list = []
        # preprocessed_data=Preprocess(feature_vectors=data)
        pair_distance = self.get_affinity(
            data=self.feature_vectors, with_diag=True)
        sorted_pair_distance_index = np.argsort(pair_distance, axis=1)
        sorted_pair_distance_labels = np.array(
            [self.labels[i] for i in sorted_pair_distance_index])
        for class_index in classes:
            nearest_dist_other_class_index = (
                sorted_pair_distance_labels != class_index).argmax(axis=1)
            nearest_dist_index = np.array([sorted_pair_distance_index[row, nearest_index]
                                           for row, nearest_index in enumerate(nearest_dist_other_class_index)])
            # print(nearest_dist_index.shape)
            nearest_dist = np.array([pair_distance[row, nearest_index]
                                     for row, nearest_index in enumerate(nearest_dist_index)])
            nearest_dist_per_class = nearest_dist[self.labels == class_index]
            plot_list.append(nearest_dist_per_class)
        self.hist(plot_list, title='compare_nearest_dist_other_class',
                  row=2, col=2)


class Cluster:
    def __init__(self, n_clusters, feature_vectors):
        self.n_clusters = n_clusters
        self.feature_vectors = feature_vectors

    def kmeans(self):
        self.model = KMeans(n_clusters=self.n_clusters)

    def kmedoids(self):
        self.model = KMedoids(n_clusters=self.n_clusters)

    def agglomerative(self, linkage, affinity):
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters, linkage=linkage, affinity=affinity)

    def birch(self):
        #acc is 0.87
        self.model = Birch(n_clusters=self.n_clusters)

    def spectral(self, affinity, n_neighbors):
        self.model = SpectralClustering(
            n_clusters=self.n_clusters, affinity =affinity, n_neighbors=n_neighbors)

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

    def goodness(self, true_labels):
        self.fit_model()
        # evaluate performance
        normalized_mutual_info = normalized_mutual_info_score(
            true_labels, self.predicted_labels)
        points = (normalized_mutual_info-0.8)/0.02 + 1
        print('current project can get {:d} points'.format(int(points)))
        return normalized_mutual_info


def try_agglomerative_params(cluster, labels):
    #result: ward and euclidean
    linkages = ["ward", "complete", "average", "single"]
    affinitys = ["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"]
    best_result = 0
    best_params = {'linkage': '', 'affinity': ''}
    for linkage, affinity in product(linkages, affinitys):
        try:
            cluster.agglomerative(linkage=linkage, affinity=affinity)
            result = cluster.goodness(labels)
            if result > best_result:
                best_result = result
                best_params['linkage'] = linkage
                best_params['affinity'] = affinity
        except Exception as error:
            print(error)
            continue
    print('result is coming')
    print(best_params)
    print(best_result)


if __name__ == "__main__":
    data = LoadData(genedata_path='../data/genedata.csv',
                    msdata_path='../data/msdata.csv', first_feature_index=2)
    gene_cluster = Cluster(n_clusters=5, feature_vectors=data.gene_feature_vectors)
    gene_cluster.spectral(affinity = 'nearest_neighbors', n_neighbors=6)
    print(gene_cluster.goodness(data.gene_labels))

    preprocess_ms = Preprocess(feature_vectors=data.ms_feature_vectors)
    # preprocess_ms.normalize()
    ratio, reduced_ms = preprocess_ms.pca(n_components=500)
    weighted_ms = reduced_ms*np.sqrt(ratio)
    # ms_affinity=preprocess_ms.get_affinity(metric='manhattan')
    print(weighted_ms.shape)
    comparison = Comparison(reduced_ms, data.ms_labels,
                            metric='euclidean', x_range=[0, 400])

    comparison.compare_pair_dist(isShow=True)
    comparison.compare_nearest_dist()
    comparison.compare_between_class_dist()
    comparison.compare_nearest_dist_other_class()
    # hist(data.ms_feature_vectors,data.ms_labels,plot_nearest=True)
    # ms_cluster = Cluster(n_clusters=3, feature_vectors=weighted_ms)
    # ms_cluster.kmeans()
    # print(ms_cluster.goodness(data.ms_labels))
