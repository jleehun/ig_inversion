import os 
import numpy as np 

if not os.path.exists("results"):
    os.makedirs("results")
attrs = np.random.random(size=(50000,224,224))
np.save('results/attrs.npy', attrs)