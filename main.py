import matplotlib.pyplot as plt
import numpy as np
import torch
from lassonet.interfaces import LassoNetIntervalRegressor, LassoNetIntervalRegressorCV

from generate import generate

# Number of individuals to be simulated
n = 3000
# Number of variables
p = 100
X, y = generate(n, p)

model = LassoNetIntervalRegressor(hidden_dims=(10, 10))


oracle, order, wrong, paths, prob = model.stability_selection(X, y, n_models=20)


def pi_thr(x, F):
    return np.minimum(1, 0.5 * ((1 - x) ** 2 / F + 1))


plt.figure(dpi=300)
plt.xlabel("Average proportion of removed variables")
plt.ylabel("Selection probability")
plt.plot([1 - s.mean() for s in prob], prob)
thresholds = np.linspace(0, 1, 1000)
for F in [0.01, 0.05, 0.1]:
    plt.plot(thresholds, pi_thr(thresholds, F), "k--", label=f"F = {F:.0%}")
plt.legend()
plt.show()


plt.figure(dpi=300)
plt.yscale("log")
plt.plot(torch.arange(len(wrong)) + 1, wrong, ".-")
plt.xlabel("Number of selected variables $q$")
plt.ylabel("Average number of wrongly selected variables $E(V)$")
plt.show()

##########################################
# data splits
##########################################
d1 = n // 3
d2 = n // 3
X1, y1 = X[:d1], y[:d1]
X2, y2 = X[d1 : d1 + d2], y[d1 : d1 + d2]
X3, y3 = X[d1 + d2 :,], y[d1 + d2 :]
X12, y12 = X[: d1 + d2], y[: d1 + d2]

##########################################
# Reproduce section 2.4 (sample-splitting)
##########################################


model = LassoNetIntervalRegressor(hidden_dims=(10, 10))
oracle1, order1, *_ = model.stability_selection(X1, y1, n_models=20)
features1 = order1[:oracle1]
print("Features selected by stability selection =", features1)

# fit dense model on selected features
model.fit(X2[:, features1], y2, dense=True)

print("C-index for sample-splitting =", model.score(X3[:, features1], y3))

##########################################
# Without sample-splitting
##########################################
model = LassoNetIntervalRegressor(hidden_dims=(10, 10))
oracle12, order12, *_ = model.stability_selection(X12, y12, n_models=20)
features12 = order12[:oracle12]
print("Features selected by stability selection =", features12)
model.fit(X12[:, features12], y12, dense=True)
print("C-index without sample-splitting =", model.score(X3[:, features12], y3))


##########################################
# Cross-validation
##########################################
cv_model = LassoNetIntervalRegressorCV(hidden_dims=(10, 10))
cv_model.fit(X12, y12)
print("C-index for cross-validation =", cv_model.score(X3, y3))
