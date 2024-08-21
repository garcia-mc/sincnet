import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from lassonet.interfaces import LassoNetIntervalRegressor
from lassonet.utils import selection_probability
from tqdm import tqdm

from generate import generate

# Number of individuals to be simulated
n = 2000
# Number of variables
p = 100
X, y = generate(n, p)


# TODO: refactor and integrate stability selection into LassoNet
def make_path(model, X, y, lambda_seq=None):
    n = len(X)
    shuffle = list(range(n))
    random.shuffle(shuffle)
    train_ind = shuffle[n // 2 : n]
    return model.path(X[train_ind], y[train_ind], lambda_seq=lambda_seq)


model = LassoNetIntervalRegressor(hidden_dims=(10, 10))
n_models = 20

path = make_path(model, X, y)
lambda_seq = [it.lambda_ for it in path]
paths = [make_path(model, X, y, lambda_seq) for _ in tqdm(range(n_models))]

# TODO: stack in selection_probability
prob = torch.stack(selection_probability(paths))


def pi_thr(x, F):
    return np.minimum(1, 0.5 * ((1 - x) ** 2 / F + 1))


def min_f(prob):
    # smallest F such that there is a value prob[j, i] such that pi_thr(1 - s[j].mean(), F) < prob[j, i]
    # min_F exists j such that pi_thr(1 - prob[j].mean(), F) < prob[j, i]
    #  0.5 * ((1 - (1 - prob[j].mean())) ** 2 / F + 1) < prob[j, i]
    # min_F exists j such that ((prob[j].mean()) ** 2 / (2 * prob[j, i] - 1) < F
    # F = argmin_j ((prob[j].mean()) ** 2 / (2 * prob[j, i] - 1)
    f_values = (prob.mean(dim=1, keepdim=True)) ** 2 / (2 * prob - 1)
    f_values[prob <= 0.5] = float("inf")
    return f_values.min(axis=0).values.sort()


plt.figure(dpi=300)
plt.xlabel("Average proportion of removed variables")
plt.ylabel("Selection probability")
plt.plot([1 - s.mean() for s in prob], prob)
thresholds = np.linspace(0, 1, 100)
for F in [0.01, 0.05, 0.1]:
    plt.plot(thresholds, pi_thr(thresholds, F), "k--", label=f"F = {F:.0%}")
plt.legend()
plt.show()


min_f_values = min_f(prob)

plt.figure(dpi=300)
plt.yscale("log")
plt.plot(torch.arange(len(min_f_values.values)) + 1, p * min_f_values.values, ".-")
plt.xlabel("Number of selected variables $q$")
plt.ylabel("Average number of wrongly selected variables $E(V)$")
plt.show()
