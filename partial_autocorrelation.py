import numpy as np

def phi_to_pacf(phi: np.ndarray) -> np.ndarray:
    """
    Map AR coefficients phi_1,...,phi_p to partial autocorrelations
    kappa_1,...,kappa_p using the backward Levinson recursion.

    Parameters
    ----------
    phi : array_like, shape (p,)
        AR(p) coefficients.

    Returns
    -------
    kappa : ndarray, shape (p,)
        Partial autocorrelations / reflection coefficients.
    """
    phi = np.asarray(phi, dtype=float)
    p = len(phi)
    if p == 0:
        return np.array([], dtype=float)

    phi_m = phi.copy()
    kappa = np.empty(p, dtype=float)

    for m in range(p, 0, -1):
        # reflection coefficient at order m
        kappa[m - 1] = phi_m[m - 1]

        if m == 1:
            break

        phi_prev = np.empty(m - 1, dtype=float)
        denom = 1.0 - kappa[m - 1] ** 2

        for k in range(m - 1):
            # backward recursion:
            # phi_k^(m-1) = (phi_k^(m) - kappa_m * phi_{m-k}^(m)) / (1 - kappa_m^2)
            phi_prev[k] = (phi_m[k] - kappa[m - 1] * phi_m[m - 2 - k]) / denom

        phi_m = phi_prev

    return kappa


def pacf_to_phi(kappa: np.ndarray) -> np.ndarray:
    """
    Map partial autocorrelations kappa_1,...,kappa_p to AR coefficients
    phi_1,...,phi_p using the forward Levinson recursion.

    Parameters
    ----------
    kappa : array_like, shape (p,)
        Partial autocorrelations / reflection coefficients.

    Returns
    -------
    phi : ndarray, shape (p,)
        AR(p) coefficients.
    """
    kappa = np.asarray(kappa, dtype=float)
    p = len(kappa)
    if p == 0:
        return np.array([], dtype=float)

    phi_prev = np.array([], dtype=float)

    for m in range(1, p + 1):
        phi_m = np.empty(m, dtype=float)
        phi_m[m - 1] = kappa[m - 1]

        for k in range(m - 1):
            # forward recursion:
            # phi_k^(m) = phi_k^(m-1) + kappa_m * phi_{m-k}^(m-1)
            phi_m[k] = phi_prev[k] + kappa[m - 1] * phi_prev[m - 2 - k]

        phi_prev = phi_m

    return phi_prev


def check_reparameterisation(phi_true):
    phi_true = np.asarray(phi_true, dtype=float)
    kappa = phi_to_pacf(phi_true)
    phi_back = pacf_to_phi(kappa)

    print("phi_true:   ", phi_true)
    print("kappa (PACF):", kappa)
    print("phi_back:  ", phi_back)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # pick a stable AR(3) by choosing kappa in (-1,1) then mapping to phi
    kappa_example = np.array([0.6341, -0.3324254, 0.42])
    phi_example = pacf_to_phi(kappa_example)

    print("Example from chosen kappa:")
    check_reparameterisation(phi_exam
