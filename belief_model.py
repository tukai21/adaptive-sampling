import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from gridspace import grab_highRes_points
from map_truth import plot_dict, feature_stats


class BeliefModel:
    def __init__(self, kernel=None, optimize=True, plot_dict=plot_dict, gpFlag=True,n_neighbors = 8):
        if optimize:
            if gpFlag:
                self.model = GaussianProcessClassifier(kernel, warm_start=True)
            else:
                self.model = KNeighborsClassifier(n_neighbors = n_neighbors)
        else:
            if gpFlag:
                self.model = GaussianProcessClassifier(kernel, warm_start=True, optimizer=None)
            else:
                self.model = KNeighborsClassifier(n_neighbors = n_neighbors)
        self.n_neighbors = n_neighbors       
        self.gpFlag = gpFlag
        self.X_train = None
        self.y_train = None
        self.is_trained = False
        self.plot_ready = False

        self.plot_dict = plot_dict
        self.legend_handles = []
        for color, label in self.plot_dict['feature_name_colors']:
            self.legend_handles.append(Patch(color=color, label=label))

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.model.fit(X, y)
        if not self.is_trained:
            self.is_trained = True
        self.plot_ready = False

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def print_kernel(self):
        if self.gpFlag:
            for kernel_list in self.model.kernel_.get_params().values():
                print("Kernel values for each feature")
                for kernel in kernel_list:
                    print(kernel)
        else:
            print("K-Neighbors Classifier")
            print("n_neighbors = " + str(self.n_neighbors))
                
    def _format_data(self):
        x_min, x_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
        y_min, y_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1

        self.xx, self.yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                       np.arange(y_min, y_max, 0.1))
        self.likelihood = self.model.predict_proba(np.c_[self.xx.ravel(), self.yy.ravel()])
        self.all_limits = (x_min, x_max, y_min, y_max)
        self.plot_ready = True

    def _format_plot(self, title):
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(self.xx.min(), self.xx.max())
        plt.ylim(self.yy.min(), self.yy.max())
        if self.gpFlag:
            plt.title("%s, LML: %.3f" %
                      (title, self.model.log_marginal_likelihood(self.model.kernel_.theta)))
        else:
            plt.title(title)

    def _format_axes(self, ax):
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(self.xx.min(), self.xx.max())
        ax.set_ylim(self.yy.min(), self.yy.max())
        ax.set_aspect('equal')
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax

    def plot(self, title='Gaussian Process', plot_types=('belief', 'entropy', 'science'), feature_stats=feature_stats, fig_id=1,
             show_train=True, lines=np.array([]), points=np.array([]), star_point=np.array([])):

        if not self.plot_ready:
            self._format_data()

        num_axes = len(plot_types)
        if num_axes == 3:
            fig = plt.figure(fig_id, figsize=(20, 5))
        elif num_axes == 2:
            fig = plt.figure(fig_id, figsize=(14, 5))
            # fig = plt.figure(fig_id)
        else:
            fig = plt.figure(fig_id, figsize=(6, 5))

        axes = {}
        for i, p_type in enumerate(plot_types):
            if p_type == 'belief':
                fig, axes['belief'] = self._plot_belief(fig, num_axes, i+1, show_train, lines)
            elif p_type == 'entropy':
                fig, axes['entropy'] = self._plot_entropy(fig, num_axes, i+1, points, star_point)
            elif p_type == 'science' and feature_stats:
                fig, axes['science'] = self._plot_science(fig, num_axes, i+1, points, star_point, feature_stats)

        if self.gpFlag:
            title = "%s, LML: %.3f" % (title, self.model.log_marginal_likelihood(self.model.kernel_.theta))

        fig.suptitle(title)
        plt.show(fig)

    def _plot_belief(self, fig, num_axes, i, show_train, lines):
        ax = fig.add_subplot(1, num_axes, i)

        if self.likelihood.shape[1] == 2:
            # Append on a blue channel (all equal to zero)
            Z = np.append(self.likelihood, np.zeros((self.likelihood.shape[0], 1)), axis=1)
            # Put the result into a color plot
            Z = Z.reshape((self.xx.shape[0], self.xx.shape[1], Z.shape[1] + 1))
            ax.imshow(Z, extent=self.all_limits, origin="lower")
            # Shrink current axis by 20%
            # box = ax.get_position()
            # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            legendHandlesList = [self.legend_handles[0], self.legend_handles[1]]
            ax.legend(handles=legendHandlesList, loc='center left', bbox_to_anchor=(1, 0.5))

        else:
            # Put the result into a color plot
            Z = self.likelihood.reshape((self.xx.shape[0], self.xx.shape[1], self.likelihood.shape[1]))
            ax.imshow(Z, extent=self.all_limits, origin="lower")
            # Shrink current axis by 20%
            # box = ax.get_position()
            # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax.legend(handles=self.legend_handles, loc='lower left', bbox_to_anchor=(1, 0.6))

            # Plot also the training points
        if show_train:
            ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=np.array(["r", "g", "b"])[self.y_train],
                       edgecolors=(0, 0, 0))

        # plot paths on the GP
        if lines.any():
            ax.plot(lines[:, 0], lines[:, 1], c="k", lineWidth=5)

        ax.set_title("Belief Map")
        ax = self._format_axes(ax)

        return fig, ax

    def _plot_entropy(self, fig, num_axes, i, points, star_point):
        ax = fig.add_subplot(1, num_axes, i)

        if self.gpFlag:
            entropy = np.sum(- self.likelihood * np.log2(self.likelihood), axis=1)
        else:

            entropy = np.zeros(self.likelihood.shape[0])

            for probs_ind in range(self.likelihood.shape[0]):
                for el in self.likelihood[probs_ind, :]:
                    if np.isclose(el, 0.0):
                        # we should be adding 0 times positive infinity, which is 0 by convention of entropy
                        entropy[probs_ind] += 0
                    else:
                        entropy[probs_ind] += -el * np.log2(el)

        # Put the result into a color plot
        Z = entropy.reshape((self.xx.shape[0], self.xx.shape[1]))
        contour = ax.contourf(self.xx, self.yy, Z, extent=self.all_limits, origin="lower")

        cbar = fig.colorbar(contour, ax=ax)
        cbar.ax.set_ylabel('Entropy')

        # plot points (if exist)
        if points.any():
            ax.scatter(points[:, 0], points[:, 1], c='k')
            if star_point.any():
                ax.scatter(star_point[0], star_point[1], c='red', marker='*', s=500)

        ax.set_title("Entropy Map")
        ax = self._format_axes(ax)

        return fig, ax

    def _plot_science(self, fig, num_axes, i, points, star_point, feature_stats):
        ax = fig.add_subplot(1, num_axes, i)

        Z = self.model.predict_proba(np.c_[self.xx.ravel(), self.yy.ravel()])
        Z_science = np.zeros(Z.shape[0])

        # this assumes we that feature_stats has the mean in the first element of every pair
        # would need to reformulate this for UCB method
        for probs_ind in range(Z.shape[0]):
            for k in range(len(feature_stats)):
                Z_science[probs_ind] += feature_stats[k][0] * Z[probs_ind, k]

        # Put the result into a color plot
        Z = Z_science.reshape((self.xx.shape[0], self.xx.shape[1]))
        contour = ax.contourf(self.xx, self.yy, Z, extent=self.all_limits, origin="lower")

        # plot points (if inputed)
        if points.any():
            ax.scatter(points[:, 0], points[:, 1], c='k')
            if star_point.any():
                ax.scatter(star_point[0], star_point[1], c='red', marker='*', s=500)

        # plot
        cbar = fig.colorbar(contour, ax=ax)
        cbar.ax.set_ylabel('Expected Science Return (mean only)')

        ax.set_title("Science Reward Map")
        ax = self._format_axes(ax)

        return fig, ax

    def plot_belief(self, title="A GP belief with Training Data", figNumber=1, plotData=True, linePlot=np.array([])):
        #assert self.is_trained, "The GP model needs to be trained to plot a belief map"

        if not self.plot_ready:
            self._format_data()

        fig = plt.figure(figNumber, figsize=(6, 5))
        ax = fig.add_subplot(1, 1, 1)

        if self.likelihood.shape[1] == 2:
            # Append on a blue channel (all equal to zero)
            Z = np.append(self.likelihood, np.zeros((self.likelihood.shape[0], 1)), axis=1)
            # Put the result into a color plot
            Z = Z.reshape((self.xx.shape[0], self.xx.shape[1], Z.shape[1] + 1))
            ax.imshow(Z, extent=self.all_limits, origin="lower")
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            legendHandlesList = [self.legend_handles[0], self.legend_handles[1]]
            ax.legend(handles=legendHandlesList, loc='center left', bbox_to_anchor=(1, 0.5))

        else:
            # Put the result into a color plot
            Z = self.likelihood.reshape((self.xx.shape[0], self.xx.shape[1], self.likelihood.shape[1]))
            ax.imshow(Z, extent=self.all_limits, origin="lower")
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax.legend(handles=self.legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))

            # Plot also the training points
        if plotData:
            ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=np.array(["r", "g", "b"])[self.y_train],
                       edgecolors=(0, 0, 0))

        # plot paths on the GP
        if linePlot.any():
            ax.plot(linePlot[:, 0], linePlot[:, 1], c="k", lineWidth=5)

        self._format_plot(title)
        
        plt.show()

        return fig, ax

    def plot_entropy(self, figNumber=2, title="A GP's entropy", points_to_plot=np.array([]), star_point=np.array([])):
        #assert self.is_trained, "The GP model needs to be trained to plot a belief map"

        if not self.plot_ready:
            self._format_data()

        fig = plt.figure(figNumber, figsize=(6, 5))
        ax = fig.add_subplot(1, 1, 1)
        
        if self.gpFlag:
            entropy = np.sum(- self.likelihood * np.log2(self.likelihood), axis=1)
        else:
            
            entropy = np.zeros(self.likelihood.shape[0])

            for probs_ind in range(self.likelihood.shape[0]):
                for el in self.likelihood[probs_ind,:]:
                    if np.isclose(el, 0.0):
                        # we should be adding 0 times positive infinity, which is 0 by convention of entropy
                        entropy[probs_ind] += 0
                    else:
                        entropy[probs_ind] += -el*np.log2(el)


        # Put the result into a color plot
        Z = entropy.reshape((self.xx.shape[0], self.xx.shape[1]))
        contourPlot = ax.contourf(self.xx, self.yy, Z, extent=self.all_limits, origin="lower")

        # plot points (if inputed)
        if points_to_plot.any():
            ax.scatter(points_to_plot[:, 0], points_to_plot[:, 1], c='k')
            if star_point.any():
                ax.scatter(star_point[0], star_point[1], c='red', marker='*', s=500)

        self._format_plot(title)

        # plot
        cbar = fig.colorbar(contourPlot)
        cbar.ax.set_ylabel('Entropy')
        
        plt.show()
        
        return fig, ax

    def plot_science(self, feature_stats, figNumber=3, title="A GP's Expected Science Gain",
                     points_to_plot=np.array([]), star_point=np.array([])):
       # assert self.is_trained, "The GP model needs to be trained to plot a belief map"

        if not self.plot_ready:
            self._format_data()

        fig = plt.figure(figNumber, figsize=(6, 5))
        ax = fig.add_subplot(1, 1, 1)
        Z = self.model.predict_proba(np.c_[self.xx.ravel(), self.yy.ravel()])

        Z_science = np.zeros(Z.shape[0])

        # this assumes we that feature_stats has the mean in the first element of every pair and that it is all we care about
        # would need to reformulate this for UCB method
        for probs_ind in range(Z.shape[0]):
            for k in range(len(feature_stats)):
                Z_science[probs_ind] += feature_stats[k][0] * Z[probs_ind, k]

        # Put the result into a color plot
        Z = Z_science.reshape((self.xx.shape[0], self.xx.shape[1]))
        contourPlot = ax.contourf(self.xx, self.yy, Z, extent=self.all_limits, origin="lower")

        # plot points (if inputed)
        if points_to_plot.any():
            ax.scatter(points_to_plot[:, 0], points_to_plot[:, 1], c='k')
            if star_point.any():
                ax.scatter(star_point[0], star_point[1], c='red', marker='*', s=500)

        self._format_plot(title)

        # plot
        cbar = fig.colorbar(contourPlot)
        cbar.ax.set_ylabel('Expected Science Return (mean only)')
        
        plt.show()
        return fig, ax


def compare_models(model_1, model_2, feature_dict,add_new_random_points=True, num_new_samples = 20):
    if add_new_random_points:
        if num_new_samples > 50:
            print('Number of new samples greater than 50, capping at 50')
            num_new_samples  =50
        elif num_new_samples < 1:
            print('Number of new samples less than 1, setting to 1')
            num_new_samples = 1
            
       # np.random.seed(0)
        
        X_highRes, y_highRes = grab_highRes_points(feature_dict)
        sample_inds = np.random.randint(0,X_highRes.shape[0],size=num_new_samples)
        new_sample = (X_highRes[sample_inds,:],y_highRes[sample_inds])

        # Assumes the two models are trained on the same initial dataset
        X_new = model_1.X_train.copy()
        y_new = model_1.y_train.copy()
        
        for new_ind in range(len(new_sample[0])):
            if new_ind == 0:
                new_points = np.array([new_sample[0][new_ind]])
            else:
                new_points = np.append(new_points,[new_sample[0][new_ind]], axis=0)
                
            X_new = np.append(X_new,[new_sample[0][new_ind]],axis=0)
            y_new = np.append(y_new,[new_sample[1][new_ind]],axis=0)
        
        model_1.fit(X_new, y_new)
        model_2.fit(X_new, y_new)
        
        model_1.plot(title='Model_1', plot_types=('belief', 'entropy'), fig_id=1, points=new_points)
        model_2.plot(title='Model_2', plot_types=('belief', 'entropy'), fig_id=1, points=new_points)
    else:
        model_1.plot(title='Model_1', plot_types=('belief', 'entropy'), fig_id=1)
        model_2.plot(title='Model_2', plot_types=('belief', 'entropy'), fig_id=1)
