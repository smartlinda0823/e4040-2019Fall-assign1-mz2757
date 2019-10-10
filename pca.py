import time
import numpy as np

def pca_naive(X, K):
    """
    PCA -- naive version

    Inputs:
    - X: (float) A numpy array of shape (N, D) where N is the number of samples,
         D is the number of features
    - K: (int) indicates the number of features you are going to keep after
         dimensionality reduction

    Returns a tuple of:
    - P: (float) A numpy array of shape (K, D), representing the top K
         principal components
    - T: (float) A numpy vector of length K, showing the score of each
         component vector
    """

    ###############################################
    # TODO: Implement PCA by extracting        #
    # eigenvector.You may need to sort the      #
    # eigenvalues to get the top K of them.     #
    ###############################################
    ###############################################
    #          START OF YOUR CODE         #
    ###############################################
    #mean of each feature
    n_samples, n_features = X.shape
    mean=np.array([np.mean(X[:,i]) for i in range(n_features)])
    #normalization
    norm_X=X-mean
    #scatter matrix
    scatter_matrix=np.dot(np.transpose(norm_X),norm_X)
    #Calculate the eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(scatter_matrix) #Calculate eig
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
    # sort eig_vec based on eig_val from highest to lowest
    eig_pairs.sort(reverse=True)
    # select the top k eig_vec
    P=np.array([ele[1] for ele in eig_pairs[:K]])
    #get new data
    data=np.dot(norm_X,np.transpose(P))
    T=np.array([ele[0] for ele in eig_pairs[:K]])
    
    #average = np.mean(X,axis=0) 
    #m, n = np.shape(X)
    #data_adjust = []
    #avgs = np.tile(average, (m, 1))
    #data_adjust = X - avgs #qujunzhi [N*D]
    #covX = np.cov(X.T)
    #eigval, eigvec = np.linalg.eig(covX)
    ##eig_pairs = [(np.abs(eigval[i]), eigvec[:,i]) for i in range(n)]
    #index = np.argsort(eigval)[::-1]
    #selectVec = np.matrix(eigvec.T[index[:K]])#select the largest row out?[D*K]
    #P = selectVec.T
    #T=eigval
    ##T=np.array([ele[0] for ele in eig_pairs[:K]])
   
    ###############################################
    #           END OF YOUR CODE         #
    ###############################################
    
    return (P, T)
