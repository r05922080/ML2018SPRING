import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from skimage import io, filters
import os
import sys
DIR = sys.argv[1]

data = []
for image in sorted(os.listdir(DIR)):
    data.append(io.imread(DIR+image).flatten())
data = np.array(data).astype("float32")
U, S, V = np.linalg.svd((data-np.mean(data, 0)).T, full_matrices=False)
W = np.dot(data-np.mean(data, 0), U)
img = int(sys.argv[2].replace(".jpg",""))
recon = np.mean(data, 0) + np.dot(W[30, :4], U[:, :4].T)
recon = (recon - np.min(recon)) / np.max(recon)

recon = (recon * 255).astype(np.uint8)
X = np.reshape(recon, (600,600,3))
plt.imsave("reconstruction.png", arr=X.astype('uint8'))
