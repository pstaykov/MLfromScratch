# k-means
K-means clustering is an unsupervised learning algorithm. It works by trying to assign 
observations to groups in such a way that members of the same group (called clusters) 
are as similar as possible to each other and as different as possible from members of other groups. 
The algorithm works as follows:

1. Choose the number of clusters (k) and randomly initialize the centroids of the clusters.
2. Assign each observation to the cluster with the nearest centroid.
3. Calculate the new centroids by taking the average of all observations assigned to that cluster. (calculating the geometric center)
4. Repeat steps 2 and 3 until the centroids do not change significantly or a certain number of iterations is reached.

We calculate the geometric center of a cluster by taking the average x and y coordinates of all points in the cluster.
