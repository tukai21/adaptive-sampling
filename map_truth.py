from matplotlib.patches import Ellipse
from gridspace import gen_gridded_space_DET, plot_sampled_features


# Global constants
feature_scale = 2
feature_colors = [(0,1,0), (0,0,1)]
simple_feature_dict = {
    1: (feature_colors[0], (Ellipse((-1.5, -1), 2.5, 6, 20),)), 
    2: (feature_colors[1], (Ellipse((1, 2), 8, 2, 170), Ellipse((2, -1), 6, 2, 60),))
}
feature_name_colors = (('Red', 'No Features'), ('Green', 'Feature 1'), ('Blue', 'Feature 2'))


complex_feature_dict = {1:(feature_colors[0], (
    Ellipse((0,0),1*feature_scale,2*feature_scale,0),
    Ellipse((2,2),1*feature_scale,2*feature_scale,30),
    Ellipse((0.5,-2),1*feature_scale,0.5*feature_scale,245),
    Ellipse((-2,-2),1*feature_scale,3*feature_scale,30),
    Ellipse((-3,2.5),2*feature_scale,0.5*feature_scale,0))), 
                   2: (feature_colors[1], (
                       Ellipse((-2,1),3*feature_scale,0.5*feature_scale,15),
                       Ellipse((0,3),1*feature_scale,1*feature_scale,15),
                       Ellipse((2,-1),4*feature_scale,1*feature_scale,70),
                       Ellipse((2.5,-2.5),2*feature_scale,0.5*feature_scale,0)))}

plot_dict = {
    'feature_name_colors': feature_name_colors
}

feature_stats = [(0, 1), (6, 4), (5, 9)]



def show_grid(N=10,feature_dict = complex_feature_dict):
    X, Y, Z = gen_gridded_space_DET(feature_dict, N=N)

    plot_sampled_features(Z, feature_dict, N=N, titleStr='Sampled Points for All Features - Low Resolution')
    