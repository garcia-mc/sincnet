import numpy as np
from scipy.stats import multivariate_normal as mvn


def nonlin_gen(mean_vector, sigma):
    """Generate synthetic data for Cox Proportional Hazards model with nonlinearity."""

    n_samples = len(mean_vector)  # Number of samples
    n_checkups = 20  # Number of checkup times
    prob_next_checkup = 0.5  # Probability of attending next checkup

    # Initialize arrays
    latent_time_u = np.zeros(n_samples)
    latent_time_v = np.zeros(n_samples)
    delta_1 = np.zeros(n_samples)
    delta_2 = np.zeros(n_samples)
    delta_3 = np.zeros(n_samples)

    # Generate latent event times
    epsilon = np.random.normal(loc=0, scale=1, size=n_samples)
    latent_times = np.exp(mean_vector + sigma * epsilon)

    for i in range(n_samples):
        # Generate random checkup times
        checkup_times = np.sort(
            np.exp(
                np.random.uniform(
                    low=np.log(np.quantile(latent_times, 0.01)),
                    high=np.log(np.quantile(latent_times, 0.99)),
                    size=n_checkups,
                )
            )
        )

        # Generate attendance to checkups
        attend_flags = np.random.rand(n_checkups) < prob_next_checkup
        attend_flags[0] = True

        attended_before_event = attend_flags & (checkup_times < latent_times[i])
        attended_after_event = attend_flags & (latent_times[i] <= checkup_times)

        # Determine latent times u and v based on attendance and event time
        if np.any(attended_before_event):
            latent_time_u[i] = checkup_times[np.max(np.nonzero(attended_before_event))]
        else:
            delta_1[i] = 1
            latent_time_u[i] = 1e-5

        if np.any(attended_after_event):
            latent_time_v[i] = checkup_times[np.min(np.nonzero(attended_after_event))]
        else:
            delta_3[i] = 1
            latent_time_v[i] = 9999

        delta_2[i] = 1 - delta_1[i] - delta_3[i]

        if latent_time_v[i] < latent_time_u[i]:
            latent_time_u[i], latent_time_v[i] = latent_time_v[i], latent_time_u[i]

    return np.column_stack(
        (latent_time_u, latent_time_v, delta_1, delta_2, delta_3, latent_times)
    )


def generate(n, p):
    rho = 0.3
    cov = np.fromfunction(lambda i, j: ((-1) ** (i - j)) * rho ** np.abs(i - j), (p, p))

    X = mvn.rvs(mean=[0] * (p), cov=cov, size=n)

    m = (
        X[:, 0]
        + X[:, 0] ** 2
        - 0.5 * (X[:, 1] + X[:, 1] ** 2)
        + 0.5 * (X[:, 2] + X[:, 2] ** 2)
        + 1
    )
    sigma = 0.5

    data = nonlin_gen(m, sigma)

    y = data.round(5)

    return X, y
