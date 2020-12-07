import matplotlib.pyplot as plt
import numpy as np
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

