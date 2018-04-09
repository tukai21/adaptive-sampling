import numpy as np
from scipy import linalg
from collections import deque
from sklearn.neighbors import KDTree
from belief_model import BeliefModel


class AdaptiveSampling:
    def __init__(self, world_map, science_map, belief_model, init_pose, feature_stats, budget, horizon=3.0, beta=0.5):
        self.world_map = world_map
        self.science_map = science_map
        self.model = belief_model
        self.pose = init_pose
        self.feature_stats = feature_stats

        # sampling hyper-parameters
        self.beta = beta
        self.horizon = horizon
        self.distance_budget = budget['distance']
        self.sampling_budget = budget['sampling']

        self.total_reward = 0
        self.points_traveled = deque([self.pose])

    def start_explore(self):
        while self.distance_budget > 0 and self.sampling_budget > 0:
            # sample next point
            next_point, reward = self.pick_next_point()
            self.total_reward += reward
            self.distance_budget -= self.movement_cost(next_point)
            self.sampling_budget -= 1
            self.pose = next_point
            self.points_traveled.append(next_point)

            # update belief model
            feature = self._query_feature(next_point)
            next_sample = (next_point, feature)
            self.update_belief(next_sample)

    def update_belief(self, new_sample):
        X_new = np.concatenate([self.model.X_train, new_sample[0].reshape(1, -1)], axis=0)
        y_new = np.concatenate([self.model.y_train, new_sample[1]], axis=0)

        self.model.fit(X_new, y_new)

        return X_new, y_new

    def sample_only_exploit(self, points):
        likelihoods = self.model.predict_proba(points)
        means = np.sum(likelihoods * np.array(self.feature_stats)[:, 0], axis=1)

        best_idx = np.argmax(means)
        best_sample_loc = points[best_idx]
        best_mean = np.max(means)

        return best_sample_loc, best_mean

    def sample_only_explore(self, points):
        likelihoods = self.model.predict_proba(points)
        entropies = np.sum(-likelihoods * np.log2(likelihoods), axis=1)

        best_idx = np.argmax(entropies)
        best_sample_loc = points[best_idx]
        highest_entropy = np.max(entropies)

        return best_sample_loc, highest_entropy

    def sample_explore_exploit(self, points):
        likelihoods = self.model.predict_proba(points)
        means = np.sum(likelihoods * np.array(self.feature_stats)[:, 0], axis=1)
        entropies = np.sum(-likelihoods * np.log2(likelihoods), axis=1)

        rewards = (1 - self.beta) * means / means.max() + self.beta * entropies / entropies.max()

        best_idx = np.argmax(rewards)
        best_sample_loc = points[best_idx]
        max_reward = np.max(rewards)

        return best_sample_loc, max_reward

    def movement_cost(self, point):
        return linalg.norm(self.pose - point)

    def pick_next_point(self):
        available_points = self._get_points_from_horizon(self.world_map)
        next_point, reward = self.sample_explore_exploit(available_points)
        return next_point, reward

    def _get_points_from_horizon(self, points):
        look_horizon = min([self.distance_budget, self.horizon])
        points_around = points[np.all(points != self.pose, axis=1)]
        tree = KDTree(points_around)
        ind = tree.query_radius(self.pose.reshape([1, -1]), r=look_horizon)[0]
        ind = sorted(ind)
        return points_around[ind]

    def _get_location_stats(self, point):
        point = point.reshape(1, -1)
        likelihoods = self.model.predict_proba(point)[0]
        mean = 0
        var = 0

        for i, f in enumerate(self.feature_stats):
            mean += f[0] * likelihoods[i]
            var += f[1] * likelihoods[i]

        return mean, var

    def _query_feature(self, point):
        feature = self.science_map[np.all(self.world_map == point, axis=1)]
        return feature

