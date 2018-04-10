import numpy as np
from numpy.testing import assert_almost_equal
import operator

from matplotlib.patches import Ellipse
from sklearn.gaussian_process.kernels import RBF
from belief_model import BeliefModel, compare_models
from gridspace import gen_gridded_space_DET, parse_map_for_GP
from sampler import AdaptiveSampling


# Global constants
feature_scale = 2
feature_colors = [(0, 1, 0), (0, 0, 1)]
feature_dict = {
    1: (feature_colors[0], (Ellipse((-1.5, -1), 2.5, 6, 20),)),
    2: (feature_colors[1], (Ellipse((1, 2), 8, 2, 170), Ellipse((2, -1), 6, 2, 60),))
}
feature_name_colors = (('Red', 'No Features'), ('Green', 'Feature 1'), ('Blue', 'Feature 2'))

plot_dict = {
    'feature_name_colors': feature_name_colors
}

feature_stats = [(0, 1), (6, 4), (5, 9)]


def test_belief(model, title, N=5, gpFlag=True, plotChoice="belief" ):
    X, Y, Z = gen_gridded_space_DET(feature_dict, N=N)
    X_train, y_train = parse_map_for_GP(X, Y, Z)
    
    model.fit(X_train, y_train)
    if plotChoice == "entropy":
        model.plot(plot_types=('belief', 'entropy'),title=title)

    elif plotChoice == "belief":
        model.plot(plot_types=('belief'), title=title)
    else:
        model.plot(plot_types=('belief', 'entropy'),title=title)
        return
        

def test_ok():
    """ If execution gets to this point, print out a happy message """
    try:
        from IPython.display import display_html
        display_html("""<div class="alert alert-success">
        <strong>Tests passed!!</strong>
        </div>""", raw=True)
    except:
        print("Tests passed!!")


def init_test(SamplerClass):
    # RECOMMEND WE CHANGE to N=
    X, Y, Z = gen_gridded_space_DET(feature_dict, N=5)
    X_init, y_init = parse_map_for_GP(X, Y, Z)
    X, Y, Z = gen_gridded_space_DET(feature_dict, N=50)
    world_map, science_map = parse_map_for_GP(X, Y, Z)

    kernel = 1.0 * RBF([1.0, 1.0])
    optimize = True
    model = BeliefModel(kernel, optimize, plot_dict)
    model.fit(X_init, y_init)

    init_pose = np.array([0, 0])
    budget = {'distance': 50, 'sampling': 10}

    sampler = SamplerClass(world_map, science_map, model, init_pose, feature_stats, budget, horizon=3)
    inds = np.array([1871, 2496, 99, 2008, 755, 797, 659, 423, 639, 544,
                     714, 2292, 151, 1207, 2076, 802, 2176, 2176, 1956, 1925])
    points = world_map[inds]

    return sampler, points


def test_exploitation(SamplerClass):
    sampler, points = init_test(SamplerClass)
    sampler_ref, _ = init_test(AdaptiveSampling)
    best_sample_loc, best_mean = sampler.sample_only_exploit(points)
    best_sample_loc_ref, best_mean_ref = sampler_ref.sample_only_exploit(points)

    assert best_sample_loc.shape == (2, ), "Your best_sample_loc has a wrong shape %s, which is expected to be (2, )." % best_sample_loc.shape
    assert isinstance(best_mean, float), "Your best_mean is not a scalar."
    assert_almost_equal(best_mean, best_mean_ref, decimal=4, err_msg="Your best_mean has a wrong value.")
    assert_almost_equal(best_sample_loc, best_sample_loc_ref, decimal=4, err_msg="Your best_sample_loc has a wrong value.")

    test_ok()
    sampler.model.plot(plot_types=('belief', 'science'), title="Selected Best Science Point (star)",
                       points=np.array(points), star_point=np.array(best_sample_loc), feature_stats=feature_stats)


def test_exploration(SamplerClass):
    sampler, points = init_test(SamplerClass)
    sampler_ref, _ = init_test(AdaptiveSampling)

    best_sample_loc, max_entropy = sampler.sample_only_explore(points)
    best_sample_loc_ref, max_entropy_ref = sampler_ref.sample_only_explore(points)
    
    assert best_sample_loc.shape == (2, ), "Your best_sample_loc has a wrong shape %s, which is expected to be (2, )." % best_sample_loc.shape
    assert isinstance(max_entropy, float), "Your max_entropy is not a scalar."
    assert_almost_equal(max_entropy, max_entropy_ref, decimal=4, err_msg="Your max_entropy has a wrong value.")
    assert_almost_equal(best_sample_loc, best_sample_loc_ref, decimal=4, err_msg="Your best_sample_loc has a wrong value.")

    test_ok()
    sampler.model.plot(plot_types=('belief', 'entropy'), title="Selected Best Explore Point (star)",
                       points=np.array(points), star_point=np.array(best_sample_loc), feature_stats=feature_stats)


def test_exploration_exploitation(SamplerClass):
    sampler, points = init_test(SamplerClass)
    sampler_ref, _ = init_test(AdaptiveSampling)

    best_sample_loc, max_reward = sampler.sample_explore_exploit(points)
    best_sample_loc_ref, max_reward_ref = sampler_ref.sample_explore_exploit(points)
    
    assert best_sample_loc.shape == (2, ), "Your best_sample_loc has a wrong shape %s, which is expected to be (2, )." % best_sample_loc.shape
    assert isinstance(max_reward, float), "Your max_reward is not a scalar."
    assert_almost_equal(max_reward, max_reward_ref, decimal=4, err_msg="Your max_reward has a wrong value.")
    assert_almost_equal(best_sample_loc, best_sample_loc_ref, decimal=4, err_msg="Your best_sample_loc has a wrong value.")

    test_ok()
    sampler.model.plot(plot_types=('belief', 'entropy', 'science'), title="Selected Best Explore-Exploit Point (star)",
                       points=np.array(points), star_point=np.array(best_sample_loc), feature_stats=feature_stats)


def test_update_belief(SamplerClass):
    sampler, points = init_test(SamplerClass)
    sampler_ref, _ = init_test(AdaptiveSampling)
    
    sampler.model.plot(plot_types=('belief', 'entropy'), title="Initial Belief Model")
    test_point = np.array([0, 1]).reshape(1, -1)

    new_point = sampler.world_map[125]
    feature = sampler.science_map[125]
    new_sample = (new_point, np.array([feature]))
    X_new, y_new = sampler.update_belief(new_sample)
    X_new_ref, y_new_ref = sampler_ref.update_belief(new_sample)

    assert X_new.shape == X_new_ref.shape, "X_new has a wrong shape %s." % X_new.shape
    assert y_new.shape == y_new_ref.shape, "y_new has a wrong shape %s." % y_new.shape

    msg = """
    Your X_new was wrongly updated, please check 
    if you had properly appended a new sample to the previous samples.
    """
    assert_almost_equal(X_new[-1], X_new_ref[-1], err_msg=msg)

    msg = """
    Your y_new was wrongly updated, please check 
    if you had properly appended a new sample to the previous samples.
    """
    assert_almost_equal(y_new[-1], y_new_ref[-1], err_msg=msg)

    new_belief = sampler.model.predict_proba(test_point)[0]
    new_belief_ref = sampler_ref.model.predict_proba(test_point)[0]

    msg = """Your updated belief is different from expected belief. 
    Please make sure you don't make any modifications to the BaseSampler class.
    """
    assert_almost_equal(new_belief, new_belief_ref, err_msg=msg)

    test_ok()
    sampler.model.plot(plot_types=('belief', 'entropy'), title="Updated Belief Model")


def test_movement_cost(SamplerClass):
    sampler, points = init_test(SamplerClass)
    sampler_ref, _ = init_test(AdaptiveSampling)

    test_points = [np.array([1, 2]), np.array([-2, 0.5]), np.array([1.5, -1])]
    for point in test_points:
        cost = sampler.movement_cost(point)
        cost_ref = sampler_ref.movement_cost(point)
        assert isinstance(cost, float), "Your cost is not a scalar."
        assert_almost_equal(cost, cost_ref, err_msg="Your computed cost is wrong.")

    test_ok()


def test_pick_next_point(SamplerClass):
    sampler, points = init_test(SamplerClass)
    sampler_ref, _ = init_test(AdaptiveSampling)
    next_point, reward = sampler.pick_next_point()
    next_point_ref, reward_ref = sampler_ref.pick_next_point()

    msg = """
    You picked up a wrong point. Please check your `sample_explore_exploit` method 
    and your usage of `_get_points_from_horizon` method.
    """
    try:
        assert_almost_equal(next_point, next_point_ref[0], err_msg=msg)
    except AssertionError:
        assert_almost_equal(next_point, next_point_ref[1], err_msg=msg)
    assert_almost_equal(reward, reward_ref, err_msg=msg)

    test_ok()


def test_start_explore(SamplerClass):
    sampler, points = init_test(SamplerClass)
    sampler_ref, _ = init_test(SamplerClass)
    ref_budget = sampler.distance_budget
    ref_point = np.array(sampler.pose)
    sampler.start_explore()

    msg = """
    Your total reward is wrong. 
    Please make sure you increment the total_reward in the sampling iteration.
    """
    assert sampler.total_reward != 0, msg

    msg = """
    Your remaining distance budget is wrong. 
    Please make sure you decrease the distance_budget by movement_cost() in the sampling iteration.
    """
    assert sampler.distance_budget != ref_budget, msg

    msg = """
    Your remaining sampling budget is wrong. 
    Please make sure you decrement the sampling_budget in the sampling iteration.
    """
    assert sampler.sampling_budget == 0, msg

    msg = """
    Your points_traveled has a wrong number of points.
    Please make sure you properly append each sampled point in the sampling iteration.
    """
    assert len(sampler.points_traveled) != 1, msg

    msg = """
    Your points_traveled has a wrong number of points.
    Please make sure you properly append each sampled point in the sampling iteration.
    """
    assert_almost_equal(sampler.points_traveled[0], ref_point, err_msg=msg)

    test_ok()
