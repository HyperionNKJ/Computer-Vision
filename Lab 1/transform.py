import numpy as np
from skimage import io
import os.path as osp

def load_image(file_name):
    """
    Load image from disk
    :param file_name:
    :return: image: numpy.ndarray
    """
    if not osp.exists(file_name):
        print('{} not exist'.format(file_name))
        return
    image = np.asarray(io.imread(file_name))
    if len(image.shape)==3 and image.shape[2]>3:
        image = image[:, :, :3]
    # print(image.shape) #should be (x, x, 3)
    return image

def save_image(image, file_name):
    """
    Save image to disk
    :param image: numpy.ndarray
    :param file_name:
    :return:
    """
    io.imsave(file_name,image)

def cs4243_resize(image, new_width, new_height):
    """
    5 points
    Implement the algorithm of nearest neighbor interpolation for image resize,
    Please round down the value to its nearest interger, 
    and take care of the order of image dimension.
    :param image: ndarray
    :param new_width: int
    :param new_height: int
    :return: new_image: numpy.ndarray
    """
    new_image = np.zeros((new_height, new_width, 3), dtype='uint8')
    if len(image.shape)==2:
        new_image = np.zeros((new_height, new_width), dtype='uint8')
    old_width = image.shape[1]
    old_height = image.shape[0]
    x_ratio = int((old_width<<16)/new_width)+1;
    y_ratio = int((old_height<<16)/new_height)+1;
    for i in range(new_height):
        for j in range(new_width):
            x2 = ((j*x_ratio)>>16)
            y2 = ((i*y_ratio)>>16)
            new_image[i][j] = image[y2][x2]
    return new_image


def cs4243_rgb2grey(image):
    """
    5 points
    Implement the rgb2grey function, use the
    weights for different channel: (R,G,B)=(0.299, 0.587, 0.114)
    Please scale the value to [0,1] by dividing 255
    :param image: numpy.ndarray
    :return: grey_image: numpy.ndarray
    """
    if len(image.shape) != 3:
        print('RGB Image should have 3 channels')
        return
    new_image = np.zeros(image.shape[:-1])
    rgb_weights = [0.299, 0.587, 0.114]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i][j]
            new_image[i][j] = pixel.dot(rgb_weights)
    image = new_image
    return image/255.


def cs4243_histnorm(image, grey_level=256):
    """
    5 points 
    Stretch the intensity value to [0, 255]
    :param image : ndarray
    :param grey_level
    :return res_image: hist-normed image
    Tips: use linear normalization here https://en.wikipedia.org/wiki/Normalization_(image_processing)
    """
    res_image = image.copy()
    res_image = res_image.astype('float64') 
    min_value = np.min(image)
    max_value = np.max(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            res_image[i][j] = (image[i][j]-min_value)*255.0/(max_value-min_value)
    return res_image


def cs4243_histequ(image, grey_level=256):
    """
    10 points
    Apply histogram equalization to enhance the image.
    the cumulative histogram will aso be returned and used in the subsequent histogram matching function.
    :param image: numpy.ndarray(float64)
    :return: ori_hist: histogram of original image
    :return: cum_hist: cumulated hist of original image, pls normalize it with image size.
    :return: res_image: image after being applied histogram equalization.
    :return: uni_hist: histogram of the enhanced image.
    Tips: use numpy buildin funcs to ease your work on image statistics
    """
    ori_hist = np.bincount(image.flatten(), minlength=grey_level)
    cum_hist = np.cumsum(ori_hist/(image.shape[0]*image.shape[1]))
    uniform_hist = cum_hist*255

    # Set the intensity of the pixel in the raw image to its corresponding new intensity 
    height, width = image.shape
    res_image = np.zeros(image.shape, dtype='uint8')  # Note the type of elements
    for i in range(height):
        for j in range(width):
            res_image[i,j] = uniform_hist[image[i,j]]
    
    uni_hist = np.bincount(res_image.flatten(), minlength=grey_level)
    return ori_hist, cum_hist, res_image, uni_hist
 
def cs4243_histmatch(ori_image, refer_image):
    """
    10 points
    Map value according to the difference between cumulative histogram.
    Note that the cum_hists of the two images can be very different. It is possible
    that a given value cum_hist[i] != cum_hist[j] for all j in [0,255]. In this case, please
    map to the closest value instead. if there are multiple intensities meet the requirement,
    choose the smallest one.
    :param ori_image #image to be processed
    :param refer_image #image of target gray histogram 
    :return: ori_hist: histogram of original image
    :return: ref_hist: histogram of reference image
    :return: res_image: image after being applied histogram matching.
    :return: res_hist: histogram of the enhanced image.
    Tips: use cs4243_histequ to help you
    """
    
    ref_hist,n,_,_ = cs4243_histequ(refer_image)
    ori_hist,k,_,_ = cs4243_histequ(ori_image)
    map_value = [np.abs(n-k[j]).argmin() for j in range(256)]
    
    # Set the intensity of the pixel in the raw image to its corresponding new intensity      
    height, width = ori_image.shape
    res_image = np.zeros(ori_image.shape, dtype='uint8')  # Note the type of elements
    for i in range(height):
        for j in range(width):
            res_image[i,j] = map_value[ori_image[i,j]]
    
    res_hist = np.bincount(res_image.flatten(), minlength=256)
    
    return ori_hist, ref_hist, res_image, res_hist


def cs4243_rotate180(kernel):
    """
    Rotate the matrix by 180. 
    Can utilize build-in Funcs in numpy to ease your work
    :param kernel:
    :return:
    """
    kernel = np.flip(np.flip(kernel, 0),1)
    return kernel

def cs4243_gaussian_kernel(ksize, sigma):
    """
    5 points
    Implement the simplified Gaussian kernel below:
    k(x,y)=exp(((x-x_mean)^2+(y-y_mean)^2)/(-2sigma^2))
    Make Gaussian kernel be central symmentry by moving the 
    origin point of the coordinate system from the top-left
    to the center. Please round down the mean value. In this assignment,
    we define the center point (cp) of even-size kernel to be the same as that of the nearest
    (larger) odd size kernel, e.g., cp(4) to be same with cp(5).
    :param ksize: int
    :param sigma: float
    :return kernel: numpy.ndarray of shape (ksize, ksize)
    """
    
    kernel = np.zeros((ksize, ksize))
    start = -int(ksize/2)
    end = -start
    x_range = np.arange(start, end+1)
    y_range = np.flip(np.copy(x_range))
    coord = np.reshape([(x,y) for y in y_range for x in x_range], (end*2+1, end*2+1, 2))
    
    if (ksize%2 == 0):
        x_range = x_range[:-1]
        y_range = y_range[:-1]
        coord = coord[:-1,:-1]

    x_mean = x_range.mean()
    y_mean = y_range.mean()
    gaussian = lambda x,y : np.exp(((x-x_mean)**2+(y-y_mean)**2)/(-2*sigma**2))
    
    for x in range(ksize):
        for y in range(ksize):
            kernel[x][y] = gaussian(coord[x][y][0], coord[x][y][1])

    return kernel / kernel.sum()

def cs4243_filter(image, kernel):
    """
    10 points
    Implement the convolution operation in a naive 4 nested for-loops,
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return:
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    filtered_image = np.zeros((Hi, Wi))
    
    ###Your code here####
    kernel = cs4243_rotate180(kernel)
    padded_image = pad_zeros(image,1,1)
    for h in range(Hi):
        for w in range(Wi):
            for r in range(Hk):
                for c in range(Wk):
                    filtered_image[h][w] = filtered_image[h][w] + padded_image[h+r][w+c] * kernel[r][c]
    ###

    return filtered_image

def pad_zeros(image, pad_height, pad_width):
    """
    Pad the image with zero pixels, e.g., given matrix [[1]] with pad_height=1 and pad_width=2, obtains:
    [[0 0 0 0 0]
    [0 0 1 0 0]
    [0 0 0 0 0]]
    :param image: numpy.ndarray
    :param pad_height: int
    :param pad_width: int
    :return padded_image: numpy.ndarray
    """
    height, width = image.shape
    new_height, new_width = height+pad_height*2, width+pad_width*2
    padded_image = np.zeros((new_height, new_width))
    padded_image[pad_height:new_height-pad_height, pad_width:new_width-pad_width] = image
    return padded_image

def cs4243_filter_fast(image, kernel):
    """
    10 points
    Implement a fast version of filtering algorithm.
    take advantage of matrix operation in python to replace the 
    inner 2-nested for loops in filter function.
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return filtered_image: numpy.ndarray
    Tips: You may find the functions pad_zeros() and cs4243_rotate180() useful
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    filtered_image = np.zeros((Hi, Wi))

    kernel = cs4243_rotate180(kernel)
    padded_image = pad_zeros(image,1,1)
    for h in range(Hi):
        for w in range(Wi):
            filtered_image[h][w] = (kernel*padded_image[h:h+Hk, w:w+Wk]).sum()

    return filtered_image

def cs4243_filter_faster(image, kernel):
    """
    10 points
    Implement a faster version of filtering algorithm.
    Pre-extract all the regions of kernel size,
    and obtain a matrix of shape (Hi*Wi, Hk*Wk),also reshape the flipped
    kernel to be of shape (Hk*Wk, 1), then do matrix multiplication, and rehshape back
    to get the final output image.
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return filtered_image: numpy.ndarray
    Tips: You may find the functions pad_zeros() and cs4243_rotate180() useful
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape

    ###Your code here####
    kernel = cs4243_rotate180(kernel).reshape((Hk*Wk, 1))
    padded_image = pad_zeros(image,int((Hk-1)/2),int((Wk-1)/2))
    regions = np.zeros((Hi*Wi, Hk*Wk))
    for h in range(Hi):
        for w in range(Wi):
            regions[h*Wi+w] = padded_image[h:h+Hk, w:w+Wk].flatten()
    filtered_image = regions.dot(kernel).reshape(image.shape)
    return filtered_image

def cs4243_downsample(image, ratio):
    """
    Downsample the image to its 1/(ratio^2),which means downsample the width to 1/ratio, and the height 1/ratio.
    for example:
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    B = downsample(A, 2)
    B=[[1, 3], [7, 9]]
    :param image:numpy.ndarray
    :param ratio:int
    :return:
    """
    width, height = image.shape[1], image.shape[0]
    return image[0:height:ratio, 0:width:ratio]

def cs4243_upsample(image, ratio):
    """
    upsample the image to its 2^ratio, 
    :param image: image to be upsampled
    :param kernel: use same kernel to get approximate value for additional pixels
    :param ratio: which means upsample the width to ratio*width, and height to ratio*height
    :return res_image: upsampled image
    """
    width, height = image.shape[1], image.shape[0]
    new_width, new_height = width*ratio, height*ratio
    res_image = np.zeros((new_height, new_width))
    res_image[0:new_height:ratio, 0:new_width:ratio] = image
    return res_image


def cs4243_gauss_pyramid(image, n=3):
    """
    10 points
    build a Gaussian Pyramid of level n
    :param image: original grey scaled image
    :param n: level of pyramid
    :return pyramid: list, with list[0] corresponding to original image.
	:e.g., img0->blur&downsample->img1->blur&downsample->img2	
    Tips: you may need to call cs4243_gaussian_kernel() and cs4243_filter_faster()
	The kernel for blur is given, do not change it.
    """
    kernel = cs4243_gaussian_kernel(7, 1)
    pyramid = []
    ## your code here####
    for i in range(n+1):
        pyramid.append(image.copy())
        image = cs4243_filter_faster(image, kernel)
        image = image[::2,::2]
    ##
    return pyramid

def cs4243_lap_pyramid(gauss_pyramid):
    """
    10 points
    build a Laplacian Pyramid from the corresponding Gaussian Pyramid
    :param gauss_pyramid: list, results of cs4243_gauss_pyramid
    :return lap_pyramid: list, with list[0] corresponding to image at level n-1 in Gaussian Pyramid.
	Tips: The kernel for blurring during upsampling is given, you need to scale its value following the standard pipeline in laplacian pyramid.
    """
    #use same Gaussian kernel 

    kernel = cs4243_gaussian_kernel(7, 1)
    n = len(gauss_pyramid)
    lap_pyramid = [gauss_pyramid[n-1]] # the top layer is same as Gaussian Pyramid
    
    for i in range(n-1):
        upsized = cs4243_upsample(gauss_pyramid[n-i-1], 2)
        upsampled = cs4243_filter_faster(upsized, kernel*4)
        residual = gauss_pyramid[n-i-2]-upsampled
        lap_pyramid.append(residual)
    
    return lap_pyramid
    
def cs4243_Lap_blend(A, B, mask):
    """
    10 points
    blend image with Laplacian pyramid
    :param A: image on the left
    :param B: image on the right
    :param mask: mask [0, 1]
    :return blended_image: same size as input image
    Tips: use cs4243_gauss_pyramid() & cs4243_lap_pyramid() to help you
    """
    kernel = cs4243_gaussian_kernel(7, 1)
    blended_image = None
    
    gaussian_A = cs4243_gauss_pyramid(A)
    gaussian_B = cs4243_gauss_pyramid(B)
    laplacian_A = cs4243_lap_pyramid(gaussian_A)
    laplacian_B = cs4243_lap_pyramid(gaussian_B) 
    
    G_R = cs4243_gauss_pyramid(mask)
    n = len(laplacian_A)
    LS = [G_R[n-1-i]*laplacian_A[i] + (1-G_R[n-1-i])*laplacian_B[i] for i in range(n)]

    image = LS[0]
    for i in range(len(LS)-1):
        upsized = cs4243_upsample(image, 2)
        upsampled = cs4243_filter_faster(upsized, kernel*4)
        image = upsampled + LS[i+1]
    blended_image = image    
    
    return blended_image