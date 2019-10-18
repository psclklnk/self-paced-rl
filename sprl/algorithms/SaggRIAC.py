# The following code was obtained from https://github.com/florensacc/rllab-curriculum
# and only slightly modified to fit the project structure. The original license of
# the software is the following:

# The MIT License (MIT)

# Copyright (c) 2016 rllab contributors

# rllab uses a shared copyright model: each contributor holds copyright over
# their contributions to rllab. The project versioning records all such
# contribution and copyright details.
# By contributing to the rllab repository through pull-request, comment,
# or otherwise, the contributor releases their content to the license and
# copyright terms herein.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random
import operator
import numpy as np

from matplotlib import pyplot as plt
import pylab
import matplotlib.colorbar as cbar
import matplotlib.patches as patches


class Region(object):

    def __init__(self, min_border, max_border, max_history=500, max_goals=500, num_random_splits=50, mode3_noise=0.1):
        self.states = []
        self.competences = []

        self.min_border = min_border
        self.max_border = max_border
        self.num_goals = 0
        self.max_goals = max_goals
        self.max_history = max_history
        self.num_random_splits = num_random_splits
        self.mode3_noise = mode3_noise

    # Add this state and competence to the region.
    def add_state(self, state, competence):
        self.states.append(state)
        self.competences.append(competence)
        self.num_goals += 1

    def is_too_big(self):
        # Split this region if it has too many goals, and if some of the goals have a positive competence!
        # Otherwise, if the competences are all 0, then there is no point in splitting.
        do_split = (self.num_goals > self.max_goals)
        return do_split

    def split(self):
        region1, region2, success = self.optimal_split()
        return [region1, region2, success]

    def assign_states_to_regions(self, region1, region2):
        # Reassign all goals to one of these regions.
        for state, competence in zip(self.states, self.competences):
            if region1.contains(state):
                region1.add_state(state, competence)
            elif region2.contains(state):
                region2.add_state(state, competence)
            else:
                raise Exception("Split region; now cannot find region for state: " + str(state))

    def optimal_split(self):
        # All split scores must be >= 0
        max_split_score = -1
        max_region1 = None
        max_region2 = None

        num_dim = len(self.min_border)
        for i in range(self.num_random_splits):
            split_dim = random.randrange(num_dim)
            split_val = random.uniform(self.min_border[split_dim], self.max_border[split_dim])
            region1, region2 = self.make_regions(split_dim, split_val)
            self.assign_states_to_regions(region1, region2)
            split_score = len(region1.states) * len(region2.states) * abs(
                region1.compute_interest() - region2.compute_interest())

            if split_score > max_split_score:
                max_region1 = region1
                max_region2 = region2
                max_split_score = split_score

        if max_split_score == -1 or max_region1 is None:
            print("Problem - unable to find a good split!")
            success = False
        else:
            success = True

        return [max_region1, max_region2, success]

    def make_regions(self, split_dim, split_val):
        # For now, just perform a single split.
        region1_min = np.copy(self.min_border)
        region1_max = np.copy(self.max_border)
        region1_max[split_dim] = split_val
        region1 = Region(region1_min, region1_max, max_history=self.max_history, max_goals=self.max_goals)

        region2_min = np.copy(self.min_border)
        region2_min[split_dim] = split_val
        region2_max = np.copy(self.max_border)
        region2 = Region(region2_min, region2_max, max_history=self.max_history, max_goals=self.max_goals)

        return [region1, region2]

    # Compute the sum of the competences in a given range.
    def compute_local_measure(self, start_index, end_index):
        return np.sum(np.array(self.competences)[start_index:end_index])

    # Compute the derivative of competences.
    def compute_interest(self):
        num_states = len(self.states)

        if num_states < 2:
            return 0

        cur_history = min(self.max_history, num_states if num_states % 2 == 0 else num_states - 1)
        split_idx = int(cur_history / 2)

        old_measure = np.array(self.competences)[num_states - cur_history:num_states - split_idx]
        new_measure = np.array(self.competences)[num_states - split_idx:num_states]

        interest = abs(np.sum(new_measure) - np.sum(old_measure)) / cur_history
        return interest

    # Check whether this state is inside this region.
    def contains(self, state):
        # Check whether this state is between the borders.
        return (np.less_equal(self.min_border, state).all() and np.less_equal(state, self.max_border).all())

    def sample_uniform(self):
        state = []
        for min_val, max_val in zip(self.min_border, self.max_border):
            state.append(random.uniform(min_val, max_val))
        return state

    def sample_mode3(self):
        # Find the lowest competence goal in this region.
        min_index, min_value = min(enumerate(self.competences), key=operator.itemgetter(1))
        bad_goal = np.copy(self.states[min_index])

        # Add noise to this goal.
        bad_goal += np.random.normal(0, self.mode3_noise, len(bad_goal))
        return bad_goal.tolist()


class SaggRIAC(object):

    def __init__(self, state_size, state_range=None, state_center=None, state_bounds=None, max_history=100,
                 max_goals=500):
        self.mode1_p = 0.7  # Percent samples from high-interest regions
        self.mode2_p = 0.2  # Percent sampled from whole space
        self.mode3_p = 0.1  # Percent sampled from mode 3 (low-performing goals in high-interest regions)

        self.max_goals = max_goals

        self.regions = []

        self.state_size = state_size
        self.state_center = np.array(state_center) if state_center is not None else np.zeros(state_size)
        if state_range is not None:
            self.state_range = state_range
            self.state_bounds = np.vstack([self.state_center - self.state_range * np.ones(self.state_size),
                                           self.state_center + self.state_range * np.ones(self.state_size)])
        elif state_bounds is not None:
            self.state_bounds = np.array(state_bounds)
            self.state_range = self.state_bounds[1] - self.state_bounds[0]

        self.min_border = self.state_bounds[0]
        self.max_border = self.state_bounds[1]

        # Create a region to represent the entire space.
        self.whole_region = Region(self.min_border, self.max_border, max_history=max_history, max_goals=self.max_goals)
        self.regions.append(self.whole_region)

    # Limit this sample to the boundaries of the region.
    def limit_sample(self, sample):
        sample = list(np.clip(sample, self.min_border, self.max_border))
        return sample

    # Find the region that contains a given state.
    def find_region(self, state):
        for index, region in enumerate(self.regions):
            if region.contains(state):
                return [index, region]
        raise Exception("Cannot find state: " + str(state) + " in any region!")

    def add_accidental_states(self, states, extend_dist_rew):
        # Treat these accidental states as if we reached them with the highest competence.
        if extend_dist_rew:
            competences = np.zeros(len(states))
        else:
            competences = np.ones(len(states))
        self.add_states(states, competences)

    # Add these states and competences to our list.
    def add_states(self, states, competences):
        for state, competence in zip(states, competences):
            # Find the appropriate region for this state.
            index, region = self.find_region(state)
            # Add this state to the region.
            region.add_state(state, competence)

            # If the region contains too many goals, split it into subregions.
            if region.is_too_big():
                [region1, region2, success] = region.split()
                if success:
                    # Add the subregions and delete the original region.
                    self.regions.append(region1)
                    self.regions.append(region2)
                    del self.regions[index]

    # Sample states from the regions.
    def sample_states(self, num_samples):

        # Mode 1
        num_samples_mode1 = int(num_samples * self.mode1_p)
        samples_mode1 = self.sample_mode_1(num_samples_mode1)
        all_samples = samples_mode1

        # Mode 2
        num_samples_mode2 = int(num_samples * self.mode2_p)
        samples_mode2 = self.sample_uniform(num_samples_mode2)
        all_samples += samples_mode2

        # Mode 3
        num_samples_mode3 = int(num_samples * self.mode3_p)
        samples_mode3 = self.sample_mode_3(num_samples_mode3)
        all_samples += samples_mode3

        remainder = num_samples - num_samples_mode1 - num_samples_mode2 - num_samples_mode3
        if remainder > 0:
            remainder_samples = self.sample_mode_1(remainder)
            all_samples += remainder_samples

        return all_samples

    # Sample uniformly at random from the whole space.
    def sample_uniform(self, num_samples):
        samples = []

        for i in range(num_samples):
            state = self.whole_region.sample_uniform()
            samples.append(state)
        return samples

    # Temporary hack - just randomly pick a region to sample from.
    def sample_random_region(self, num_samples):
        samples = []
        for i in range(num_samples):
            region_index = random.randrange(len(self.regions))
            region = self.regions[region_index]
            state = region.sample_uniform()
            samples.append(state)
        return samples

    def sample_mode_3(self, num_samples):
        return self.sample_mode_1(num_samples, mode3=True)

    def sample_mode_1(self, num_samples, mode3=False):
        if len(self.regions) == 1:
            return self.sample_random_region(num_samples)

        interests = self.compute_all_interests()
        probs = interests

        if sum(probs) == 0:
            return self.sample_uniform(num_samples)

        num_per_regions = np.random.multinomial(num_samples, probs)

        samples = []
        for region_index, num_per_region in enumerate(num_per_regions):
            if num_per_region > 0:
                region = self.regions[region_index]
                for i in range(num_per_region):
                    if mode3:
                        sample = region.sample_mode3()
                        sample = self.limit_sample(sample)
                    else:
                        sample = region.sample_uniform()
                    samples.append(sample)

        return samples

    def compute_all_interests(self):
        interests = np.array([])
        for region in self.regions:
            interests = np.append(interests, region.compute_interest())

        # Subtract the min interest
        min_interest = min(interests)
        interests -= min_interest

        sum_interests = sum(interests)
        if sum_interests == 0:
            return np.zeros(len(interests))

        # Normalize
        interests /= sum_interests

        return interests

    def plot_regions_interest(self, ax = None):
        if ax is None:
            __, ax = plt.subplots()

        interests = self.compute_all_interests()
        interest_lims = (min(interests), max(interests))
        normal = pylab.Normalize(*interest_lims)

        colors = pylab.cm.YlOrRd(normal(interests))

        for region, color in zip(self.regions, colors):
            lengths = region.max_border - region.min_border
            ax.add_patch(patches.Rectangle(region.min_border, *lengths, fill=True, edgecolor='k', facecolor=color))

        cax, _ = cbar.make_axes(ax)
        print("the interest lims are: ", interest_lims)
        cb2 = cbar.ColorbarBase(cax, cmap=pylab.cm.YlOrRd, norm=normal)
        ax.set_xlim(self.state_bounds[0][0], self.state_bounds[1][0])
        ax.set_ylim(self.state_bounds[0][1], self.state_bounds[1][1])

    def plot_regions_states(self):
        fig, ax = plt.subplots()

        states_per_reg = [len(region.states) for region in self.regions]
        states_per_reg_lims = (min(states_per_reg), max(states_per_reg))
        normal = pylab.Normalize(*states_per_reg_lims)

        colors = pylab.cm.BuGn(normal(states_per_reg))

        for region, color in zip(self.regions, colors):
            lengths = region.max_border - region.min_border
            ax.add_patch(patches.Rectangle(region.min_border, *lengths, fill=True, edgecolor='k', facecolor=color))

        cax, _ = cbar.make_axes(ax)
        print("the interest lims are: ", states_per_reg_lims)
        cb2 = cbar.ColorbarBase(cax, cmap=pylab.cm.BuGn, norm=normal)
        ax.set_xlim(self.state_bounds[0][0], self.state_bounds[1][0])
        ax.set_ylim(self.state_bounds[0][1], self.state_bounds[1][1])
