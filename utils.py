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


h_default = .1  # step size in the mesh

# Plotting feature for a GP and its input data.
# TODO: breakdown into a class that then can plot entropy, feature_stats, etc

class GP_Plotter():
    def __init__(self,GP, feature_name_colors, X_train, y_train,h = h_default):
        self.GP = GP
        self.feature_name_colors = feature_name_colors
        self.X_train = X_train
        self.y_train = y_train
        self.h = h
        
        x_lim = [ X_train[:,0].min(),X_train[:,0].max() ]
        y_lim = [ X_train[:,1].min(),X_train[:,1].max() ]
        x_min, x_max = x_lim[0] - 1, x_lim[1] + 1
        y_min, y_max = y_lim[0] - 1, y_lim[1] + 1
    
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
        
        self.legendHandlesList = []
        for pair in feature_name_colors:
            self.legendHandlesList.append(matplotlib.patches.Patch(color=pair[0],label = pair[1]))
            
        self.all_limits = [x_min, x_max, y_min, y_max]
        self.xx = xx
        self.yy = yy
        
    def plot_GP_and_data(self,figNumber = 1,title = "A GP with Training Data",plotData = True,linePlot = np.array([])):
        
        
        fig = plt.figure(figNumber,figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1) 
        Z = self.GP.predict_proba(np.c_[self.xx.ravel(), self.yy.ravel()])
        num_features = Z.shape[1]
        
        # Put the result into a color plot
        Z = Z.reshape((self.xx.shape[0], self.xx.shape[1], num_features ))
        ax.imshow(Z, extent=(self.all_limits), origin="lower")

        # Plot also the training points
        if plotData:
            ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=np.array(["r", "g", "b"])[self.y_train],
                        edgecolors=(0, 0, 0))
            
        # plot paths on the GP
        if linePlot.any():
            ax.plot(linePlot[:,0],linePlot[:,1],c="k",lineWidth = 5)
            
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(self.xx.min(), self.xx.max())
        plt.ylim(self.yy.min(), self.yy.max())
        plt.title("%s, LML: %.3f" %
                  (title, self.GP.log_marginal_likelihood(self.GP.kernel_.theta)))
                   
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(handles= self.legendHandlesList,loc='center left', bbox_to_anchor=(1, 0.5)) 
        return fig,ax

    def plot_GP_entropy(self,figNumber = 2, title = "A GP's entropy",points_to_plot = np.array([]),star_point =np.array([])):
        fig = plt.figure(figNumber,figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1) 
        Z = self.GP.predict_proba(np.c_[self.xx.ravel(), self.yy.ravel()])
        num_features = Z.shape[1]
        
        
            
        Z_entropy = np.zeros(Z.shape[0])

        for probs_ind in range(Z.shape[0]):
            for el in Z[probs_ind,:]:
                Z_entropy[probs_ind] += -el*np.log2(el)
        
        # Put the result into a color plot
        Z = Z_entropy.reshape((self.xx.shape[0], self.xx.shape[1] ))
        contourPlot = ax.contourf(self.xx,self.yy,Z, extent=(self.all_limits), origin="lower")
        
        # plot points (if inputed)
        if points_to_plot.any():
            ax.scatter(points_to_plot[:,0],points_to_plot[:,1], c='k')
            if star_point.any():
                ax.scatter(star_point[0],star_point[1], c='red',marker ='*', s = 500)
                
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(self.xx.min(), self.xx.max())
        plt.ylim(self.yy.min(), self.yy.max())
        plt.title("%s, LML: %.3f" %
                  (title, self.GP.log_marginal_likelihood(self.GP.kernel_.theta)))
                   
        # plot
        cbar = fig.colorbar( contourPlot  )
        cbar.ax.set_ylabel('Entropy')
        return fig,ax
    
    def plot_GP_expected_science(self,feature_stats ,figNumber = 3, title = "A GP's Expected Science Gain",points_to_plot = np.array([]),star_point = np.array([])):
        
        fig = plt.figure(figNumber,figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1) 
        Z = self.GP.predict_proba(np.c_[self.xx.ravel(), self.yy.ravel()])
        num_features = Z.shape[1]
        
            
        Z_science = np.zeros(Z.shape[0])
        
        # this assumes we that feature_stats has the mean in the first element of every pair and that it is all we care about
        # would need to reformulate this for UCB method
        for probs_ind in range(Z.shape[0]):
            for k in range(len(feature_stats)):
                Z_science[probs_ind] += feature_stats[k][0]*Z[probs_ind,k]
        
        # Put the result into a color plot
        Z = Z_science.reshape((self.xx.shape[0], self.xx.shape[1] ))
        contourPlot = ax.contourf(self.xx,self.yy,Z, extent=(self.all_limits), origin="lower")
        
        # plot points (if inputed)
        if points_to_plot.any():
            ax.scatter(points_to_plot[:,0],points_to_plot[:,1], c='k')
            if star_point.any():
                ax.scatter(star_point[0],star_point[1], c='red',marker ='*', s = 500)
                
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(self.xx.min(), self.xx.max())
        plt.ylim(self.yy.min(), self.yy.max())
        plt.title("%s, LML: %.3f" %
                  (title, self.GP.log_marginal_likelihood(self.GP.kernel_.theta)))
                   
        # plot
        cbar = fig.colorbar( contourPlot  )
        cbar.ax.set_ylabel('Expected Science Return (mean only)')
        return fig,ax
    
            