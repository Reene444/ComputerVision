import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math

def show_rgb_image(image, title=None):
    # Converts from one colour space to the other. this is needed as RGB
    # is not the default colour space for OpenCV

    mpl.rcParams['figure.dpi'] =  300


    # Show the image
    plt.imshow(image)

    # remove the axis / ticks for a clean looking image
    plt.xticks([])
    plt.yticks([])

    # if a title is provided, show it
    if title is not None:
        plt.title(title)

    plt.show()

def show_binary_image(image, title=None):
    # Converts from one colour space to the other. this is needed as RGB
    # is not the default colour space for OpenCV

    # Show the image

    plt.imshow(image, cmap=plt.cm.gray)

    # remove the axis / ticks for a clean looking image
    plt.xticks([])
    plt.yticks([])

    # if a title is provided, show it
    if title is not None:
        plt.title(title)

    plt.show()


# Gives the Probability Density Function (pdf) of the Normal 
# Distribution with mean, Standard Deviation (sd) std_dev, for the value x.
# example: sample_gaussian(0.1,0,np.arange(-3,4,1,dtype=np.float32)));
# Calculates pdf with mean = 0, sd = 0.1, for a vector of 7 elements long
def sample_gaussian(std_dev,mean,vec):

    x= -np.square(vec-float(mean))/(2.0*math.pow(std_dev,2))
    
    return np.array([1/(std_dev * math.sqrt(2* math.pi))  * np.exp(x)])


def zero_cross(image):
    z_c_image = np.zeros(image.shape)
    thresh = np.absolute(image).mean() * 0.75
    h,w = image.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            patch = image[y-1:y+2, x-1:x+2]
            p = image[y, x]
            maxP = patch.max()
            minP = patch.min()
            if (p > 0):
                zeroCross = True if minP < 0 else False
            else:
                zeroCross = True if maxP > 0 else False
            if ((maxP - minP) > thresh) and zeroCross:
                z_c_image[y, x] = 1
    return z_c_image

