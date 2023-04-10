
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.colors import ListedColormap


def process_heatmap(R, my_cmap=plt.cm.seismic(np.arange(plt.cm.seismic.N))):
    power = 1.0
    b = 10*((np.abs(R)**power).mean()**(1.0/power))
    my_cmap[:,0:3] *= 0.99
    my_cmap = ListedColormap(my_cmap)
    return (R, {"cmap":my_cmap, "vmin":-b, "vmax":b, "interpolation":'nearest'} )