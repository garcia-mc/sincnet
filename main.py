import random

import matplotlib.pyplot as plt
import numpy as np
from lassonet.interfaces import LassoNetIntervalRegressor
from lassonet.utils import selection_probability
from tqdm import tqdm

from generate import generate


# Number of individuals to be simulated
n = 2000
# Number of variables
p = 100
X, y = generate(n, p)


def make_path(lambda_seq=None):
    model = LassoNetIntervalRegressor(hidden_dims=(10, 10))
    shuffle = list(range(n))
    random.shuffle(shuffle)
    train_ind = shuffle[n // 2 : n]
    return model.path(X[train_ind], y[train_ind], lambda_seq=lambda_seq)


n_models = 20

path = make_path()
lambda_seq = [it.lambda_ for it in path]
paths = [make_path(lambda_seq) for _ in tqdm(range(n_models))]


prob = selection_probability(paths)


def curve_family(x, F):
    return np.minimum(1, 0.5 * ((1 - x) ** 2 / F + 1))


plt.xlabel("Average proportion of removed variables")
plt.ylabel("Selection probability")
plt.plot([1 - s.mean() for s in prob], prob)
thresholds = np.linspace(0, 1, 100)
for F in [0.1, 0.2, 0.3]:
    plt.plot(thresholds, curve_family(thresholds, F), "k--", label=f"F={F}")
plt.legend()
