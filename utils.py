import sklearn
import scipy
import meta_utils
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


from matplotlib.patches import Ellipse
import matplotlib.patches 
import matplotlib.path as mpltPath

xlim_default = (-3, 3)
ylim_default = (-3, 3)
h_default = .02  # step size in the mesh

# Plotting feature for a GP and its input data
def plot_GP_and_data(GP,X_train,y_train,title = "A GP with Training Data",x_lim = xlim_default, y_lim = ylim_default, h = h_default):
    
    x_min, x_max = x_lim[0] - 1, x_lim[1] + 1
    y_min, y_max = y_lim[0] - 1, y_lim[1] + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    
    plt.figure(figsize=(10, 5))
    Z = GP.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    num_features = Z.shape[1]
    # Put the result into a color plot
    
    Z = Z.reshape((xx.shape[0], xx.shape[1], num_features ))
    plt.imshow(Z, extent=(x_min, x_max, y_min, y_max), origin="lower")

    # Plot also the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=np.array(["r", "g", "b"])[y_train],
                edgecolors=(0, 0, 0))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("%s, LML: %.3f" %
              (title, GP.log_marginal_likelihood(GP.kernel_.theta)))

    plt.tight_layout()
    plt.show()