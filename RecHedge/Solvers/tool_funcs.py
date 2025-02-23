import numpy as np


# projection to a box constraint set
def proj_box(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """
    Projection to a box constraint set
    :param x: point
    :param lb: lower bound
    :param ub: upper bound
    :return y: projection point
    """
    y = x
    sign_id_lower = x < lb
    y[sign_id_lower] = lb[sign_id_lower]
    sign_id_upper = x > ub
    y[sign_id_upper] = ub[sign_id_upper]
    return y


def proj_simplex(v: np.ndarray, bound: float = 1) -> np.ndarray:
    """
    Projection onto the simplex with the given bound.
    This is the classical algorithm, see the slides titled "Projection onto simplex" of Andersen Ang
    Link: https://angms.science/doc/CVX/Proj_simplex.pdf
    :param v: point.
    :param bound: bound of the simplex
    :return w: projection point
    """
    if bound <= 0:
        raise ValueError("Bound must be positive.")
    
    v = v.flatten()  # obtain an 1D array
    n = len(v)
    u = np.sort(v)[::-1]  # sort in descending order
    cssv = np.cumsum(u) - bound
    ind = np.arange(n) + 1
    target_id = np.where(u - cssv / ind > 0)[0][-1]  # find the index corresponding to the threshold
    rho = ind[target_id]
    theta = cssv[target_id] / rho
    w = np.maximum(v - theta, 0)  # threshold
    
    return w.reshape(-1, 1)


if __name__ == '__main__':
    # Example usage
    # v = np.array([0.2, 0.1, -0.1, 0.4, 0.5]).reshape(-1, 1)
    v = np.array([[2.16125623],
       [0.01068014],
       [0.86140965],
       [1.82994352],
       [0.23671047]])
    bound = 5
    projected_v = proj_simplex(v, bound)
    print("Original vector:", v.T)
    print("Projected vector:", projected_v.T)
    