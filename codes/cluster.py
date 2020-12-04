import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import GridSearchCV
from sklearn_extra.cluster import KMedoids
from itertools import product
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

    def pca(self, n_components):
        model = PCA(n_components=n_components)
        reduced_data = model.fit_transform(self.data)
        print(model.explained_variance_ratio_)
        return reduced_data


def visualize2D(data):
    """ 
      input: 2d Data
    """
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()


class Cluster:
    def __init__(self, n_clusters, feature_vectors):
        self.n_clusters = n_clusters
        self.feature_vectors = feature_vectors

    def kmeans(self):
        self.model = KMeans(n_clusters=self.n_clusters)

    def kmedoids(self):
        self.model = KMedoids(n_clusters=self.n_clusters)

    def agglomerative(self,linkage, affinity):
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters,linkage=linkage, affinity=affinity)

    def spectral(self):
        self.model = SpectralClustering(n_clusters=self.n_clusters)

    def goodness(self, true_labels):
        # fit model and predict
        self.model.fit(self.feature_vectors)
        self.predicted_labels = self.model.labels_
        # evaluate performance
        normalized_mutual_info = normalized_mutual_info_score(
            true_labels, self.predicted_labels)
        points = (normalized_mutual_info-0.8)/0.02 + 1
        print('current project can get {:d} points'.format(int(points)))
        return normalized_mutual_info

def try_agglomerative_params(cluster,labels):
    #result: ward and euclidean
    linkages=["ward", "complete", "average", "single"]
    affinitys=["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"]
    best_result=0
    best_params={'linkage':'','affinity':''}
    for linkage, affinity in product(linkages,affinitys):
        try:
            cluster.agglomerative(linkage=linkage, affinity=affinity)
            result=cluster.goodness(labels)
            if result>best_result:
                best_result=result
                best_params['linkage']=linkage
                best_params['affinity']=affinity
        except Exception as error:
            print(error)
            continue
    print('result is coming')
    print(best_params)
    print(best_result)


if __name__ == "__main__":
    data = LoadData(genedata_path='../data/genedata.csv',
                    msdata_path='../data/msdata.csv', first_feature_index=2)
    # print(data.gene_feature_vectors)
    preprocess_gene = Preprocess(feature_vectors=data.gene_feature_vectors)
    # normalized_gene=preprocess_gene.normalize()
    # reduced_gene = preprocess_gene.pca(n_components=20)
    cluster = Cluster(n_clusters=5, feature_vectors=data.gene_feature_vectors)
    # cluster.kmeans()
    # cluster.agglomerative()
    try_agglomerative_params(cluster,data.gene_labels)
    # print(cluster.goodness(data.gene_labels))