import random

import numpy as np
from small_text import random_initialization_balanced


def random_initialization_biased(y, n_samples=10, non_sample=None):
    """Draw half class-1 (excluding `non_sample` indices) and half class-0.

    Used to simulate a biased warm-start: the active learner is initialised with target
    examples drawn from only part of the target population. The held-out part is what
    NIHDAL's "non-seeded target" diagnostics track.
    """
    expected_samples_per_class = np.floor(n_samples / 2).astype(int)

    all_indices = [i for i, lab in enumerate(y) if lab == 1 and i not in non_sample]
    target_sample = random.sample(all_indices, expected_samples_per_class)

    all_indices = [i for i, lab in enumerate(y) if lab == 0]
    other_sample = random.sample(all_indices, expected_samples_per_class)

    return np.array(target_sample + other_sample)


def initialize_active_learner_biased(active_learner, y_train, biased_indices, n_samples=100):
    indices_initial = random_initialization_biased(
        y_train, n_samples=n_samples, non_sample=biased_indices
    )
    active_learner.initialize_data(indices_initial, y_train[indices_initial])
    return indices_initial


def initialize_active_learner_balanced(active_learner, y_train, n_samples=100):
    indices_initial = random_initialization_balanced(y_train, n_samples=n_samples)
    active_learner.initialize_data(indices_initial, y_train[indices_initial])
    return indices_initial
