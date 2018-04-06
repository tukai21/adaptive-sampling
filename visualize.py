import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
import matplotlib.patches
import matplotlib.path as mpltPath


h_default = .1  # step size in the mesh
N_default = 50
xlim_default = (-3, 3)
ylim_default = (-3, 3)


class GP_Plotter():
    def __init__(self, GP, feature_name_colors, X_train, y_train, h=h_default):
        self.GP = GP
        self.feature_name_colors = feature_name_colors
        self.X_train = X_train
        self.y_train = y_train
        self.h = h

        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        self.legendHandlesList = []
        for pair in feature_name_colors:
            self.legendHandlesList.append(matplotlib.patches.Patch(color=pair[0], label=pair[1]))

        self.all_limits = [x_min, x_max, y_min, y_max]
        self.xx = xx
        self.yy = yy

    def plot_GP_and_data(self, figNumber=1, title="A GP with Training Data", plotData=True, linePlot=np.array([])):
        fig = plt.figure(figNumber, figsize=(6, 5))
        ax = fig.add_subplot(1, 1, 1)
        Z = self.GP.predict_proba(np.c_[self.xx.ravel(), self.yy.ravel()])
        num_features = Z.shape[1]

        if num_features == 2:
            # Append on a blue channel (all equal to zero)
            Z = np.append(Z, np.zeros((Z.shape[0], 1)), axis=1)
            # Put the result into a color plot
            Z = Z.reshape((self.xx.shape[0], self.xx.shape[1], num_features + 1))
            ax.imshow(Z, extent=(self.all_limits), origin="lower")
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            legendHandlesList = [self.legendHandlesList[0], self.legendHandlesList[1]]
            ax.legend(handles=legendHandlesList, loc='center left', bbox_to_anchor=(1, 0.5))
            '''
            Z = Z[:,1].reshape((self.xx.shape[0], self.xx.shape[1]))
            contourPlot = ax.contourf(self.xx,self.yy,Z, extent=(self.all_limits), origin="lower")
            cbar = fig.colorbar( contourPlot  )
            cbar.ax.set_ylabel('Probability of Feature #1')'''

        else:
            # Put the result into a color plot
            Z = Z.reshape((self.xx.shape[0], self.xx.shape[1], num_features))
            ax.imshow(Z, extent=(self.all_limits), origin="lower")
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax.legend(handles=self.legendHandlesList, loc='center left', bbox_to_anchor=(1, 0.5))

            # Plot also the training points
        if plotData:
            ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=np.array(["r", "g", "b"])[self.y_train],
                       edgecolors=(0, 0, 0))

        # plot paths on the GP
        if linePlot.any():
            ax.plot(linePlot[:, 0], linePlot[:, 1], c="k", lineWidth=5)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(self.xx.min(), self.xx.max())
        plt.ylim(self.yy.min(), self.yy.max())
        plt.title("%s, LML: %.3f" %
                  (title, self.GP.log_marginal_likelihood(self.GP.kernel_.theta)))

        return fig, ax

    def plot_GP_entropy(self, figNumber=2, title="A GP's entropy", points_to_plot=np.array([]),
                        star_point=np.array([])):
        fig = plt.figure(figNumber, figsize=(6, 5))
        ax = fig.add_subplot(1, 1, 1)
        Z = self.GP.predict_proba(np.c_[self.xx.ravel(), self.yy.ravel()])
        num_features = Z.shape[1]

        Z_entropy = np.zeros(Z.shape[0])

        for probs_ind in range(Z.shape[0]):
            for el in Z[probs_ind, :]:
                Z_entropy[probs_ind] += -el * np.log2(el)

        # Put the result into a color plot
        Z = Z_entropy.reshape((self.xx.shape[0], self.xx.shape[1]))
        contourPlot = ax.contourf(self.xx, self.yy, Z, extent=(self.all_limits), origin="lower")

        # plot points (if inputed)
        if points_to_plot.any():
            ax.scatter(points_to_plot[:, 0], points_to_plot[:, 1], c='k')
            if star_point.any():
                ax.scatter(star_point[0], star_point[1], c='red', marker='*', s=500)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(self.xx.min(), self.xx.max())
        plt.ylim(self.yy.min(), self.yy.max())
        plt.title("%s, LML: %.3f" %
                  (title, self.GP.log_marginal_likelihood(self.GP.kernel_.theta)))

        # plot
        cbar = fig.colorbar(contourPlot)
        cbar.ax.set_ylabel('Entropy')
        return fig, ax

    def plot_GP_expected_science(self, feature_stats, figNumber=3, title="A GP's Expected Science Gain",
                                 points_to_plot=np.array([]), star_point=np.array([])):

        fig = plt.figure(figNumber, figsize=(6, 5))
        ax = fig.add_subplot(1, 1, 1)
        Z = self.GP.predict_proba(np.c_[self.xx.ravel(), self.yy.ravel()])
        num_features = Z.shape[1]

        Z_science = np.zeros(Z.shape[0])

        # this assumes we that feature_stats has the mean in the first element of every pair and that it is all we care about
        # would need to reformulate this for UCB method
        for probs_ind in range(Z.shape[0]):
            for k in range(len(feature_stats)):
                Z_science[probs_ind] += feature_stats[k][0] * Z[probs_ind, k]

        # Put the result into a color plot
        Z = Z_science.reshape((self.xx.shape[0], self.xx.shape[1]))
        contourPlot = ax.contourf(self.xx, self.yy, Z, extent=(self.all_limits), origin="lower")

        # plot points (if inputed)
        if points_to_plot.any():
            ax.scatter(points_to_plot[:, 0], points_to_plot[:, 1], c='k')
            if star_point.any():
                ax.scatter(star_point[0], star_point[1], c='red', marker='*', s=500)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(self.xx.min(), self.xx.max())
        plt.ylim(self.yy.min(), self.yy.max())
        plt.title("%s, LML: %.3f" %
                  (title, self.GP.log_marginal_likelihood(self.GP.kernel_.theta)))

        # plot
        cbar = fig.colorbar(contourPlot)
        cbar.ax.set_ylabel('Expected Science Return (mean only)')
        return fig, ax


def plot_truth_features(feature_dict, N=N_default, xlim=xlim_default, ylim=ylim_default, oneFeature=False):
    '''
    INPUT:
    -features_dict: dictionary with key of feature type with the following value:
        --Value is a tuple with the following elements (length of this tuple corresponds to the number of features of this type):
            ---First element is the color of the feature (for plotting)
            ---Second elemnt is an ellipse object
                ----((x_center,y_center), width, height, angle (in degrees))
    OUTPUT:
    -a plot showing the truth location of the features (plot resolution)'''
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    legendHandlesList = []
    for feature_num in feature_dict.keys():
        feature_color = feature_dict[feature_num][0]
        feature_ellipses = feature_dict[feature_num][1]
        legendHandlesList.append(matplotlib.patches.Patch(color=feature_color, label='Feature #: ' + str(feature_num)))
        for e in feature_ellipses:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.5)
            e.set_facecolor(feature_color)

        if oneFeature:
            break

    # create legend based on colors
    titleStr = 'Truth Plot for All Features'
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    plt.suptitle(titleStr)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(handles=legendHandlesList, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_sampled_features(X, Y, Z, feature_dict, N=N_default, xlim=xlim_default, ylim=ylim_default, plotNoFeatures=True,
                          alpha=0.9, titleStr='Sampled Points for All Features'):
    '''
    INPUT:
    -features_dict: dictionary with key of feature type with the following value:
        --Value is a tuple with the following elements (length of this tuple corresponds to the number of features of this type):
            ---First element is the color of the feature (for plotting)
            ---Second elemnt is an ellipse object
                ----((x_center,y_center), width, height, angle (in degrees))
    OUTPUT:
    -a plot showing the sampled location of the features (grid resolution)'''

    x = np.linspace(xlim[0], xlim[1], N)
    y = np.linspace(ylim[0], ylim[1], N)

    # Plot information from Z only
    # loop through each feature and plot
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

    legendHandlesList = []
    # scatter plot for no feature present
    if plotNoFeatures:
        feature_color = "red"
        legendHandlesList.append(matplotlib.patches.Patch(color=feature_color, label='No Features'))
        x_inds = []
        y_inds = []
        for point in Z.keys():
            if 0 in Z[point]:
                x_inds.append(point[1])
                y_inds.append(point[0])
        ax.scatter(x[x_inds], y[y_inds], color=feature_color, alpha=alpha / 2)

    for feature_num in feature_dict.keys():
        feature_color = feature_dict[feature_num][0]
        legendHandlesList.append(matplotlib.patches.Patch(color=feature_color, label='Feature #: ' + str(feature_num)))
        x_inds = []
        y_inds = []
        for point in Z.keys():
            if feature_num in Z[point]:
                x_inds.append(point[1])
                y_inds.append(point[0])
        ax.scatter(x[x_inds], y[y_inds], color=feature_color, alpha=alpha)

    # create legend based on colors

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    plt.suptitle(titleStr)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(handles=legendHandlesList, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()