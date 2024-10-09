"""
kmeans
The kmeans class is supposed to handle the weights of the neural network and 
cluster them according to their values. Here we decide at the beginning how many clusters we want

Also it can plot the weight distribution

Author: Anton Giese
Date: 26.10.2020
"""


from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt



class myKmeans():
    """
    init
    @param weightsData: save the weights (this is a one dimensional data array from the csc format)
    """
    def __init__(self,weightsData):
        self.weightsData = weightsData


    """
    showDistOfWeights
    This function is supposed to show how the weights in the weights matrix are distributed. 
    It plots the weights vs. the occurences (Histogramm)
    """
    def showDistOfWeights(self,data):
        plt.hist(data, bins=np.arange(data.min(), data.max()+1,0.01))
        plt.show()

    
    """
    cluster
    This is the function that actually applies kmeans
    We get the labels using the predict method (assignment of the OLD weights to the new centroids)
    We get the actual centroids with kmeans.cluster_centers_
    @param k: the k in k-means (how many clusters do you want)

    """
    def cluster(self,k):
        # get the kmeans object
        kmeans = KMeans(n_clusters=k)
        # apply the fit method
        try:
            kmeans.fit(self.weightsData)
        except:
            self.weightsData = np.reshape(self.weightsData,(-1,1))
            kmeans.fit(self.weightsData)
        
        # extract the new found centroids
        self.centroids = kmeans.cluster_centers_

        # extract the labels (the label array in the csc format for quantized data)
        self.labels = kmeans.predict(self.weightsData)
        #print("labels: ",self.labels)

        self.weightsDataQuantized = self.centroids[self.labels][:, :]  # quantised kernel
        #print("centroids:",self.centroids)
        #print("weightsdataquant",self.weightsDataQuantized)
        #print("inertia: ",kmeans.inertia_) # measure of how good the clustering is
        return self.centroids,self.labels,kmeans.inertia_,self.weightsDataQuantized

        


if __name__ == "__main__":
    data = np.array([1.2,1.6,-3.2,2.2,-0.56,-0.22,6.3,-4,4,0,0.442,-0.11])
    data = np.reshape(data,(-1,1))
    print(data)
    myK = myKmeans(data)
    #myK.showDistOfWeights(data)
    myK.cluster(5)
    x = [1,1,5,6,1,5,10,22,23,23,50,51,51,52,100,112,130,500,512,600,12000,12230]






    # old attempts using MeanShift (recommended in the internet for one dimensional data)
"""
    X = np.array([x,np.zeros(len(x))])#, dtype=np.int)
    bandwidth = 2#estimate_bandwidth(X, quantile=0.1)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    for k in range(n_clusters_):
        my_members = labels == k
        print ("cluster {0}: {1}".format(k, X[my_members, 0]))
"""