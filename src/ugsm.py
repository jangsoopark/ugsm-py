from scipy import fftpack
import numpy as np

import copy

import utils


def ugsm(image, params):
    h, w = image.shape

    c = utils.get_c(image)

    r = image
    s = np.zeros((h, w), dtype=np.float32)

    # positive regularization parameters for proposed l1-norm model in equation (15), (16), ...
    lambda_1 = params['lambda_1']
    lambda_2 = params['lambda_2']

    # regularization parameters for ALM in equation (17), ...
    beta_1 = params['beta_1']
    beta_2 = params['beta_2']
    beta_3 = params['beta_3']

    # lagrange multipliers for ALM in equation (17), ...
    p1 = np.zeros((w, h), dtype=np.float32)
    p2 = np.zeros((w, h), dtype=np.float32)
    p3 = np.zeros((w, h), dtype=np.float32)

    tol = params['tol']
    max_iter = params['max_iter']

    # left hand side coefficient in equation (25)
    denominator = beta_1 * c['dxTdx'] + beta_2 * np.ones((h, w)) + beta_3 * c['dyTdy']

    # l1-norm model in equation (15), (16)
    d_s = utils.derivative_y(s)  # D_y(D_theta (s))
    d_r_s = utils.derivative_x(r - s)  # D_x(D_theta ( r - s))

    i = 0
    relative_change = 1
    while relative_change > tol and i < max_iter:
        print(relative_change)
        # u-sub-problem in equation (19)
        _u = d_r_s + p1.T / beta_1
        u = np.sign(_u) * np.maximum(0, np.abs(_u) - lambda_1 / beta_1)

        # v-sub-problem in equation (19)
        _v = s + p2.T / beta_2
        v = np.sign(_v) * np.maximum(0, np.abs(_v) - lambda_2 / beta_2)

        # w-sub-problem in equation (19)
        _w = d_s + p3.T / beta_3
        w = np.sign(_w) * np.maximum(0, np.abs(_w) - 1 / beta_3)

        # s-sub-problem in equation (24), (25)
        _s = copy.deepcopy(s)

        t1 = beta_1 * utils.derivative_x(r) - beta_1 * u + p1.T  # beta_1 * d_x r - beta_1 * u + p_1
        t2 = beta_2 * v - p2.T  # beta_2 * v - p_2
        t3 = beta_3 * w - p3.T  # beta_3 * w - p3

        _s1 = utils.derivative_xy(t1, t3) + t2
        _s2 = fftpack.fft2(_s1) / denominator
        s = np.real(fftpack.ifft2(_s2))
        s[s < 0] = 0  # equation (26)

        index = s > r
        s[index] = r[index]

        u1 = r - s
        u2 = r - _s

        relative_change = np.linalg.norm(u1 - u2, ord='fro') / np.linalg.norm(u1, ord='fro')

        d_s = utils.derivative_y(s)
        d_r_s = utils.derivative_x(r - s)

        # update p
        p1 = p1 + 1.618 * beta_1 * (d_r_s - u).T
        p2 = p2 + 1.618 * beta_2 * (s - v).T
        p3 = p3 + 1.618 * beta_3 * (d_s - w).T
        i += 1

    return s, i
