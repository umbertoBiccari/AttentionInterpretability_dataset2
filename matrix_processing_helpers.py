import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

def sparsify_global_percentile(M, q=90, keep_diagonal=False):
    """ 
    Sparsify a matrix given a percentile

    Parameters
    ----------
    M : 2D-numpy array
        matrix to be sparsified
    q : int
        percentile --> the values below the q-percentile of all the values in M are set to zero
    keep_diagonal: bool
        if False, the values on the diagonal are set to zero

    Returns
    ----------
    M_sparse: sparsified matrix
    """
    
    M_sparse = M.copy()
    
    if keep_diagonal:
        vals = M_sparse.ravel()
    else:
        mask = ~np.eye(M_sparse.shape[0], dtype=bool)
        vals = M_sparse[mask]
        np.fill_diagonal(M_sparse, 0.0)

    tau = np.percentile(vals, q)
    M_sparse[M_sparse < tau] = 0.0
    
    return M_sparse

def sparsify_topk_rows(M, k=5, keep_diagonal=False):
    """ 
    Sparsify a matrix using a top-k rule

    Parameters
    ----------
    M : 2D-numpy array
        matrix to be sparsified
    k : int
        for each row, we keep only the top-k values
    keep_diagonal: bool
        if False, the values on the diagonal are set to zero

    Returns
    ----------
    M_sparse: sparsified matrix
    """
    
    M_sparse = np.zeros_like(M)
    d = M.shape[0]

    for i in range(d):
        row = M[i].copy()

        if not keep_diagonal:
            row[i] = -np.inf

        idx = np.argpartition(row, -k)[-k:]
        M_sparse[i, idx] = M[i, idx]

    if not keep_diagonal:
        np.fill_diagonal(M_sparse, 0.0)

    return M_sparse

def reorder(M, importance, keep_diagonal=False):
    """ 
    Reorder a matrix according to an importance vector

    Parameters
    ----------
    M : 2D-numpy array
        matrix to be reordered
    importance : 1D-numpy array
        sorted importance vector        
    keep_diagonal: bool
        if False, the values on the diagonal are set to zero

    Returns
    ----------
    M_reordered: reordered matrix
    """
        
    if not keep_diagonal:
        np.fill_diagonal(M, 0.0)

    M_reordered = M[importance][:, importance]      

    return M_reordered

def reorder_clustering(M, make_symmetric=True, keep_diagonal=False):
    """ 
    Reorder a matrix: 
        - treat each row as a feature vector
        - cluster features based on similarity
        - reorder according to the clustering tree

    Parameters
    ----------
    M : 2D-numpy array
        matrix to be reordered
    make_symmetric : bool
        if true, makes M symmetric M -> 0.5*M.M.transpose)
    keep_diagonal: bool
        if False, the values on the diagonal are set to zero

    Returns
    ----------
    M_reordered: reordered matrix
    """
        
    if not keep_diagonal:
        np.fill_diagonal(M, 0.0)

    if make_symmetric:
        M = 0.5 * (M + M.T)

    X = M   
    D = pdist(X, metric='euclidean')   
    Z = linkage(D, method='average')   
    idx = leaves_list(Z)

    M_reordered = M[idx][:, idx]      

    return M_reordered