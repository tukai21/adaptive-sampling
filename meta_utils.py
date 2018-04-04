import sklearn
import scipy
import meta_utils
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt


from matplotlib.patches import Ellipse
import matplotlib.patches 
import matplotlib.path as mpltPath

N_default = 50
xlim_default = (-3, 3)
ylim_default = (-3, 3)


def gen_gridded_space_DET(feature_dict,oneFeaturePerPoint = True,N = N_default,xlim = xlim_default,ylim = ylim_default):
    '''
    This can be used to model maps that have different features, but each feature is deterministically at a point or not
    INPUT:
    -features_dict: dictionary with key of feature type with the following value:
        --Value is a tuple with the following elements (length of this tuple corresponds to the number of features of this type):
            ---First element is the color of the feature (for plotting)
            ---Second elemnt is an ellipse object
                ----((x_center,y_center), width, height, angle (in degrees))
    -N: Integer number of grid points in each direction
    -xlim: tuple composed of x_lb and x_ub
    -ylim: tuple composed of y_lb and y_ub

    OUTPUT:
    -(X, Y, Z) tuple corresponding to a meshgrid of X, Y values and Z, which is a dict with N^2 keys and a value corresponding to a tuple that records the indicies where each feature type is present
    '''
    
    x = np.linspace(xlim[0],xlim[1],N)
    y = np.linspace(ylim[0],ylim[1],N)
    X,Y = np.meshgrid(x,y)
    # loop through each ellipse and return grid of points
    Z = {}
    for x_ind in range(len(x)):
        for y_ind in range(len(y)):
            Z[(x_ind,y_ind)] = []

    for feature_num in feature_dict.keys():
        feature_color = feature_dict[feature_num][0]
        feature_ellipses = feature_dict[feature_num][1]
        for e in feature_ellipses:
            xc = X - e.center[0]
            yc = Y - e.center[1]
            cos_angle = np.cos(np.radians(180.-e.angle))
            sin_angle = np.sin(np.radians(180.-e.angle))
            xct = xc * cos_angle - yc * sin_angle
            yct = xc * sin_angle + yc * cos_angle

            rad_cc  = (xct**2/(e.width/2.)**2) + (yct**2/(e.height/2.)**2)
            inds = np.where(rad_cc <=1)
            for k in range(len(inds[1])):
                if oneFeaturePerPoint:
                    # only one entry in the list
                    if not Z[(inds[0][k],inds[1][k])]:
                        # only append if empty
                        Z[(inds[0][k],inds[1][k])].append(feature_num)
                else:
                    Z[(inds[0][k],inds[1][k])].append(feature_num)

            ''' Another approach (more intuitive, but harder to put into a point index dictionary - Z):
            X_ellip = X[np.where(rad_cc <=1)]
            Y_ellip = Y[np.where(rad_cc <=1)]
            ax.scatter(X_ellip,Y_ellip,color=feature_color)'''
      # add zeros to blank indicies
    for x_ind in range(len(x)):
        for y_ind in range(len(y)):
            if not Z[(x_ind,y_ind)]:
                # no features
                Z[(x_ind,y_ind)] = [0]
            
    return (X,Y,Z)

def parse_map_for_GP(X,Y,Z,parseNoFeatures = False):
    '''
    INPUT: X,Y,Z from gen_gridded_space_DET
    
    OUTPUT: one tuple with X and y values formatted for as numpy arrays for input into GP tools
    '''
    N = X.shape[0]
    x_lim = [X.min(), X.max()]
    y_lim = [Y.min(),Y.max()]
    x = np.linspace(x_lim[0],x_lim[1],N)
    y = np.linspace(y_lim[0],y_lim[1],N)

    
    X_forGP = []
    y_forGP = []
    for inds in Z.keys():
        for feature in Z[inds]:
            if parseNoFeatures:
                X_forGP.append(np.array((x[inds[1]],y[inds[0]])))
                y_forGP.append(np.array(feature,dtype=int))
            else:
                if feature is not 0:
                    X_forGP.append(np.array((x[inds[1]],y[inds[0]])))
                    y_forGP.append(np.array(feature,dtype=int))

    X_forGP = np.array(X_forGP)
    y_forGP = np.array(y_forGP)
    
    return (X_forGP, y_forGP)

def plot_truth_features(feature_dict,N = N_default,xlim = xlim_default,ylim = ylim_default, oneFeature = False):
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
        legendHandlesList.append(matplotlib.patches.Patch(color=feature_color,label = 'Feature #: ' + str(feature_num)))
        for e in feature_ellipses:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.5)
            e.set_facecolor(feature_color)
        
        if oneFeature:
            break
            
    
    # create legend based on colors
    titleStr = 'Truth Plot for All Features'
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(ylim[0],ylim[1])
    plt.suptitle(titleStr)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(handles= legendHandlesList,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def plot_sampled_features(X,Y,Z,feature_dict,N = N_default,xlim = xlim_default,ylim = ylim_default,plotNoFeatures = True,alpha=0.9,titleStr = 'Sampled Points for All Features' ):
    '''
    INPUT:
    -features_dict: dictionary with key of feature type with the following value:
        --Value is a tuple with the following elements (length of this tuple corresponds to the number of features of this type):
            ---First element is the color of the feature (for plotting)
            ---Second elemnt is an ellipse object
                ----((x_center,y_center), width, height, angle (in degrees))
    OUTPUT:
    -a plot showing the sampled location of the features (grid resolution)'''
    
    x = np.linspace(xlim[0],xlim[1],N)
    y = np.linspace(ylim[0],ylim[1],N)
    
    # Plot information from Z only
    # loop through each feature and plot
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

    legendHandlesList = []
    # scatter plot for no feature present
    if plotNoFeatures:
        feature_color = "red"
        legendHandlesList.append(matplotlib.patches.Patch(color=feature_color,label = 'No Features'))
        x_inds= []
        y_inds = []
        for point in Z.keys():   
            if 0 in Z[point]:
                x_inds.append(point[1])
                y_inds.append(point[0])
        ax.scatter(x[x_inds],y[y_inds],color=feature_color,alpha=alpha/2)
        
    for feature_num in feature_dict.keys():
        feature_color = feature_dict[feature_num][0]
        legendHandlesList.append(matplotlib.patches.Patch(color=feature_color,label = 'Feature #: ' + str(feature_num)))
        x_inds= []
        y_inds = []
        for point in Z.keys():   
            if feature_num in Z[point]:
                x_inds.append(point[1])
                y_inds.append(point[0])
        ax.scatter(x[x_inds],y[y_inds],color=feature_color,alpha=alpha)

    # create legend based on colors
    
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(ylim[0],ylim[1])
    plt.suptitle(titleStr)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(handles= legendHandlesList,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
 

def gen_gridded_space_PDF(features,cov_mats,N = N_default,xlim = xlim_default,ylim = ylim_default):
    '''
    This can be used to model gradients of feature 'intensity' via the PDF 
    INPUT:
    -features: a tuple in which the index corresponds to the number of the feature and element corresponds to the number of that type of feature
    -cov_mat: 2x2 covariance matrix for each feature
    -N: Integer number of grid points in each direction
    -xlim: tuple composed of x_lb and x_ub
    -ylim: tuple composed of y_lb and y_ub

    OUTPUT:
    -(X, Y, Z) tuple corresponding to a meshgrid of X, Y values and Z, which is the pdf of finding the corresponding feature
    '''
    
    # create gridded space
    X = np.linspace(xlim[0],xlim[1],N)
    Y = np.linspace(ylim[0],ylim[1],N)
    X, Y = np.meshgrid(X, Y)
    # Pack X and Y into a single 3-dimensional array
    pos = np.zeros(X.shape + (2,))
    pos[:,:,0] = X
    pos[:,:,1] = Y
    
    num_features = len(features)
    Z_list = []
    
    for feature_ind in range(num_features):
        num_of_this_feature = features[feature_ind]
        cov_mat = cov_mats[feature_ind]
        Z = np.zeros(X.shape)
        for k in range(num_of_this_feature):
            mean_x = np.random.uniform(xlim[0],xlim[1])
            mean_y = np.random.uniform(ylim[0],ylim[1])
            # Mean vector and covariance matrix
            mu = np.array([mean_x, mean_y])
            F = multivariate_normal(mu,cov_mat)

            # add probabilities together to create final map
            Z += F.pdf(pos)

        Z_list.append(Z)
        
    return (X, Y, Z_list)