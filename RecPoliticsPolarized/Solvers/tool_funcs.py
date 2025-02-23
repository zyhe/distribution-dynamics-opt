import numpy as np


# projection to a box constraint set
def proj_box(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """
    projection to a box constraint set
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


# projection to an l2 norm ball
def proj_ball(x: np.ndarray, radi: float) -> np.ndarray:
    """
    projection to the l2 norm ball
    :param x: point
    :param radi: radius of the norm ball
    :return y: projection point
    """
    norm = np.linalg.norm(x)
    if norm <= radi:
        return x
    else:
        return radi * x / norm


# generate a positive definite matrix
def pd_mat(dim: int, seed: int) -> np.ndarray:
    """
    generate a positive definite matrix
    :param dim: number of row of the square matrix
    :param seed: number of the random seed
    :return: W: a positive definite matrix
    """
    np.random.seed(seed)
    A = 5*np.random.randn(dim, dim)
    A -= np.mean(A)
    W = A.T @ A
    rho = np.max(np.linalg.eigvals(W))
    return W / rho


# test the functions
if __name__ == '__main__':
    dim = 10
    Lambda1 = 0.3 * np.diag(np.random.rand(dim))
    Lambda3 = 0.3 * np.diag(np.random.rand(dim))
    Lambda2 = np.eye(dim) - Lambda1 - Lambda3
    W = pd_mat(dim, 10)
    mat_mid = np.eye(dim) - Lambda1 @ W
    map_inv = np.linalg.inv(mat_mid)
    ss_dec = map_inv @ Lambda2  # decision, dim * dim
    ss_init_state = map_inv @ Lambda3  # initial state, dim * dim
    pass
