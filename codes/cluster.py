import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
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
        self.data=(self.data-self.data.mean())/self.data.std()


    def pca(self, n_components):
        model = PCA(n_components=n_components)
        reduced_data=model.fit_transform(self.data)
        print(model.explained_variance_ratio_)
        return reduced_data


def visualize2D(data):
  """ 
    input: 2d Data
  """
  plt.figure()
  plt.scatter(data[:,0],data[:,1])
  plt.show()

if __name__ == "__main__":
    data = LoadData(genedata_path='../data/genedata.csv',
                    msdata_path='../data/msdata.csv', first_feature_index=2)
    # print(data.gene_feature_vectors)
    preprocess_gene=Preprocess(feature_vectors=data.gene_feature_vectors)
    preprocess_gene.normalize()
    reduced_gene=preprocess_gene.pca(n_components=10)
