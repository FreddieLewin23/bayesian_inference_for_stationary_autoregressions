import numpy as np


def pacf_to_phi(kappa):
    """
    Convert partial autocorrelations kappa_1,...,kappa_p
    to AR coefficients phi_1,...,phi_p using Levinson recursion.
    """
    kappa = np.asarray(kappa, float)
    p = len(kappa)

    phi_list = [None] * (p + 1)
    phi_list[0] = np.array([])

    for m in range(1, p + 1):
        phi_prev = phi_list[m - 1]
        phi_new = np.zeros(m)
        phi_new[-1] = kappa[m - 1]

        for j in range(m - 1):
            phi_new[j] = phi_prev[j] - kappa[m - 1] * phi_prev[m - 2 - j]

        phi_list[m] = phi_new

    return phi_list[p]


def sample_stationary_ar4(n_samples=5000):
    kappa_samples = np.random.uniform(-1, 1, size=(n_samples, 4))
    phi_samples = np.zeros_like(kappa_samples)
    for i in range(n_samples):
        phi_samples[i] = pacf_to_phi(kappa_samples[i])
    return kappa_samples, phi_samples

kappa_samps, phi_samps = sample_stationary_ar4(5000)
phi_samps.shape

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_phi = pd.DataFrame(phi_samps, columns=["phi1","phi2","phi3","phi4"])

sns.pairplot(df_phi, diag_kind="kde")
plt.suptitle("Pairs Plot of Stationary AR(4) Coefficients", y=1.02)

df_kappa = pd.DataFrame(kappa_samps, columns=["kappa1","kappa2","kappa3","kappa4"])

sns.pairplot(df_kappa, diag_kind="kde")
plt.suptitle("Pairs Plot of PACF Parameters (Uniform in (-1,1))", y=1.02)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def sample_stationary_ar4(n_samples=10000):
    phis = []

    for _ in range(n_samples):
        # Step 1: sample 4 reciprocal roots λi inside the unit disc
        # real or complex, but must pair complex values
        # simplest: sample all 4 real for demonstration
        lambdas = np.random.uniform(-1, 1, size=4)

        # Step 2: convert to polynomial roots r_i = 1/λ_i
        roots = 1.0 / lambdas

        # Step 3: construct polynomial Φ(z) = ∏ (1 - r_i z)
        # numpy.poly takes roots and returns coefficients of polynomial:
        #   z^4 - (sum r_i) z^3 + ... + (-1)^4 * prod r_i
        poly = np.poly(roots)

        # poly = [1, -phi1, -phi2, -phi3, -phi4]
        phi = -poly[1:]     # flip the signs
        phis.append(phi)

    return np.array(phis)


# Generate samples
phi_samples = sample_stationary_ar4(5000)

# Pairs plot
df = pd.DataFrame(phi_samples, columns=["phi1", "phi2", "phi3", "phi4"])

sns.pairplot(df, corner=True, plot_kws={"s": 5, "alpha": 0.4})
plt.suptitle("Stationary AR(4) Coefficients via Reciprocal-Root Parameterisation")
plt.savefig("/Users/FreddieLewin/Desktop/dissertation/plots/pairs_reciproots_ar4.png")
plt.show()
