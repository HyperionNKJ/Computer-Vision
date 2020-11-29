""" CS4243 Lab 2: Image Segmentations
See accompanying Jupyter notebook (lab2.ipynb) and PDF (lab2.pdf) for instructions.
"""
import cv2
import numpy as np
import random

from time import time
from skimage import color
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._joblib import Parallel
from sklearn.utils._joblib import delayed


# Part 1 

def smoothing(img):
    """Smooth image using Guassain filter.

    Args:
        img (np.ndarray)            : Input image of size (H, W, 3).

    Returns:
        img_smoothed (np.ndarray)   : Output smoothed image of size (H, W, 3).

    """

    """ YOUR CODE STARTS HERE """
    img_smoothed = cv2.GaussianBlur(img, (5,5), sigmaX=5.0, sigmaY=5.0)
    """ YOUR CODE ENDS HERE """

    return img_smoothed

def RGBtoLab(img):
    """Convert RGB image into L*a*b color space.

    Args:
        img (np.ndarray)            : Input RGB image  of size (H, W, 3).


    Returns:
        lab (np.ndarray)            : Converted L*a*b image of size (H, W, 3).

    """

    """ YOUR CODE STARTS HERE """
    lab = color.rgb2lab(img)
    """ YOUR CODE ENDS HERE """
   
    return lab



# Part 2
def k_means_clustering(data,k):
    """ Estimate clustering centers using k-means algorithm.

    Args:
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        k (int)                     : Number of centroids

    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)

    """
    start = time()


    """ YOUR CODE STARTS HERE """
    def initialize_random(data, k):
        copy = data.copy()
        np.random.shuffle(copy)
        return copy[:k]
    
    def initialize_zero(num):
        return np.zeros(num, dtype=int)
    
    def assign_labels(data, centers, labels):
        for i in range(len(data)):
            dists = np.array([np.linalg.norm(data[i]-center) for center in centers])
            labels[i] = np.argmin(dists) 
            
    def revise_centroids(data, labels, centers):
        for center_label in range(len(centers)):
            centers[center_label] = data[labels == center_label].mean(axis=0)
            
    def array_equal(arr1, arr2, thresh):
        return np.max(np.abs(arr1-arr2)) <= thresh
    
    data = data.astype(np.float64) # avoided discretization for accuracy
    centers = initialize_random(data, k)
    labels = initialize_zero(data.shape[0])
    stop_thresh = 0.4 # may be increased to achieve faster results
    while(True):
            old_centers = centers.copy()
            assign_labels(data, centers, labels)
            revise_centroids(data, labels, centers)
            if (array_equal(old_centers, centers, stop_thresh)): break 
    """ YOUR CODE ENDS HERE """

    end =  time()
    kmeans_runtime = end - start
    print("K-means running time: %.3fs."% kmeans_runtime)
    return labels, centers



def get_bin_seeds(data, bin_size, min_bin_freq=1):
    """ Generate initial bin seeds for windows sampling.

    Args:
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        bin_size (float)            : Bandwidth.
        min_bin_freq (int)          : For each bin_seed, number of the minimal points should cover.

    Returns:
        bin_seeds (List)            : Reprojected bin seeds. All bin seeds with total point number 
                                      bigger than the threshold.
    """

    """ YOUR CODE STARTS HERE """
    
    def compress_points(data, bin_size):
        return np.round(data/bin_size)
    
    def group_compressed(compressed):
        seeds = {}
        for pixel in compressed:
            _pixel = tuple(pixel)
            if _pixel in seeds:
                seeds[_pixel] += 1
            else:
                seeds[_pixel] = 1
        return np.array(list(seeds.keys())), np.array(list(seeds.values()))  
    
    def filter_seeds(bin_seeds, bin_freq, min_bin_freq):
        bin_seeds = bin_seeds[bin_freq >= min_bin_freq]
                
    def reproject_seeds(bin_seeds, bin_size):
        bin_seeds *= bin_size
            
    compressed = compress_points(data, bin_size)
    bin_seeds, bin_freq = group_compressed(compressed)
    filter_seeds(bin_seeds, bin_freq, min_bin_freq)
    reproject_seeds(bin_seeds, bin_size)
    
    """ YOUR CODE ENDS HERE """
    
    return bin_seeds

def mean_shift_single_seed(start_seed, data, nbrs, max_iter):
    """ Find mean-shift peak for given starting point.

    Args:
        start_seed (np.ndarray)     : Coordinate (x, y) of start seed. 
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        nbrs (class)                : Class sklearn.neighbors._unsupervised.NearestNeighbors.
        max_iter (int)              : Max iteration for mean shift.

    Returns:
        peak (tuple)                : Coordinate (x,y) of peak(center) of the attraction basin.
        n_points (int)              : Number of points in the attraction basin.
                              
    """

    # For each seed, climb gradient until convergence or max_iter
    bandwidth = nbrs.get_params()['radius']
    stop_thresh = 1e-3 * bandwidth  # when mean has converged

    """ YOUR CODE STARTS HERE """
    
    mean = start_seed
    num_iter = 0
    while(True):
        old_mean = mean
        neighbors_points = nbrs.radius_neighbors([mean])
        neighbors = data[neighbors_points[1][0]]
        mean = np.mean(neighbors, axis=0)
        num_iter += 1
        if (np.max(np.abs(old_mean - mean)) <= stop_thresh or num_iter == max_iter): break
    
    peak = tuple(mean)
    n_points = len(nbrs.radius_neighbors([mean]))
  
    """ YOUR CODE ENDS HERE """

    return peak, n_points


def mean_shift_clustering(data, bandwidth=0.7, min_bin_freq=5, max_iter=300):
    """pipline of mean shift clustering.

    Args:
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        bandwidth (float)           : Bandwidth parameter for mean shift algorithm.
        min_bin_freq(int)           : Parameter for get_bin_seeds function.
                                      For each bin_seed, number of the minimal points should cover.
        max_iter (int)              : Max iteration for mean shift.

    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)
    """
    start = time()
    n_jobs = None
    seeds = get_bin_seeds(data, bandwidth, min_bin_freq)
    n_samples, n_features = data.shape
    center_intensity_dict = {}

    # We use n_jobs=1 because this will be used in nested calls under
    # parallel calls to _mean_shift_single_seed so there is no need for
    # for further parallelism.
    nbrs = NearestNeighbors(radius=bandwidth, n_jobs=1).fit(data)
    # execute iterations on all seeds in parallel
    all_res = Parallel(n_jobs=n_jobs)(
        delayed(mean_shift_single_seed)
        (seed, data, nbrs, max_iter) for seed in seeds)

    # copy results in a dictionary
    for i in range(len(seeds)):
        if all_res[i] is not None:
            center_intensity_dict[all_res[i][0]] = all_res[i][1]

    if not center_intensity_dict:
        # nothing near seeds
        raise ValueError("No point was within bandwidth=%f of any seed."
                         " Try a different seeding strategy \
                         or increase the bandwidth."
                         % bandwidth)
    


    """ YOUR CODE STARTS HERE """
    peaks = list(center_intensity_dict.keys())
    intensities = list(center_intensity_dict.values())
    sorted_ip = sorted(zip(intensities,peaks), key=lambda pair: pair[0])
    i = 0
    while(i < len(sorted_ip)-1):
        ip = sorted_ip[i]
        j = i+1
        is_ip_deleted = False
        while (j < len(sorted_ip)):
            ip_j = sorted_ip[j]
            if (np.linalg.norm(np.array(ip_j[1]) - np.array(ip[1])) < bandwidth):
                if (ip_j[0] < ip[0]):
                    del sorted_ip[j]
                else:
                    del sorted_ip[i]
                    is_ip_deleted = True
                    break
            else:
                j+=1
        if (not is_ip_deleted):
            i+=1        
    
    labels = np.zeros(len(data), dtype=int)
    centers = np.array([peak for _,peak in sorted_ip])
    for i in range(len(data)):
            dists = np.array([np.linalg.norm(data[i]-center) for center in centers])
            labels[i] = np.argmin(dists)             

    """ YOUR CODE ENDS HERE """
    
    end =  time()
    kmeans_runtime = end - start
    print("mean shift running time: %.3fs."% kmeans_runtime)
    return labels, centers


#Part 3:

def k_means_segmentation(img, k):
    """Descrption.

    Args:
        img (np.ndarray)            : Input image of size (H, W, 3).
        k (int)                     : Number of centroids
    
    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)

    """

    """ YOUR CODE STARTS HERE """
    shape = img.shape
    depth = 1 if len(shape) == 2 else shape[2]
    labels, centers = k_means_clustering(img.reshape((shape[0]*shape[1],depth)),k)
    
    """ YOUR CODE ENDS HERE """

    return labels,centers


def mean_shift_segmentation(img,b):
    """Descrption.

    Args:
        img (np.ndarray)            : Input image of size (H, W, 3).
        b (float)                     : Bandwidth.

    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)

    """

    """ YOUR CODE STARTS HERE """
    shape = img.shape
    depth = 1 if len(shape) == 2 else shape[2]
    labels, centers = mean_shift_clustering(img.reshape((shape[0]*shape[1],depth)),bandwidth=b, min_bin_freq=1)
    
    """ YOUR CODE ENDS HERE """

    return labels, centers














"""Helper functions: You should not have to touch the following functions.
"""
def load_image(im_path):
    """Loads image and converts to RGB format

    Args:
        im_path (str): Path to image

    Returns:
        im (np.ndarray): Loaded image (H, W, 3), of type np.uint8.
    """


    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def colors(k):

    """generate the color for the plt.scatter.

    Args:
        k (int): the number of the centroids

    Returns:
        ret (list): list of colors .

    """

    colour = ["coral", "dodgerblue", "limegreen", "deeppink", "orange", "darkcyan", "rosybrown", "lightskyblue", "navy"]
    if k <= len(colour):
        ret = colour[0:k]
    else:
        ret = []
        for i in range(k):
            ret.append((random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
    return ret

def stack_seg(img, labels, centers):
    """stack segmentations for visualization.

    Args:
        img (np.ndarray): image
        labels(np.ndarray): lables for every pixel. 
        centers(np.ndarray): cluster centers.

    Returns:
        np.vstack(result) (np.ndarray): stacked result.

    """

    labels = labels.reshape((img.shape[:-1]))
    reduced = np.uint8(centers)[labels]
    result = [np.hstack([img])]
    for i, c in enumerate(centers):
        mask = cv2.inRange(labels, i, i)
        mask = np.dstack([mask]*3) # Make it 3 channel
        ex_img = cv2.bitwise_and(img, mask)
        result.append(np.hstack([ex_img]))

    return np.vstack(result)
