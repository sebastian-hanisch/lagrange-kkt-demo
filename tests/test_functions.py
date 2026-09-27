import numpy as np

import kkt_functions as fn


def test_surface_area_gradient_matches_finite_differences():
    x = np.array([1.3, 0.7])
    eps = 1e-6
    numeric = np.zeros(2)
    for i in range(2):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        numeric[i] = (fn.surface_area(xp) - fn.surface_area(xm)) / (2 * eps)
    np.testing.assert_allclose(fn.grad_surface_area(x), numeric, rtol=1e-4)


def test_volume_constraint_gradient_matches_finite_differences():
    x = np.array([1.3, 0.7])
    V0 = 10.0
    eps = 1e-6
    numeric = np.zeros(2)
    for i in range(2):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        numeric[i] = (fn.volume_constraint(xp, V0) - fn.volume_constraint(xm, V0)) / (2 * eps)
    np.testing.assert_allclose(fn.grad_volume_constraint(x), numeric, rtol=1e-4)


def test_volume_constraint_is_zero_when_volume_matches():
    r, h = 1.0, 10.0 / (np.pi * 1.0 ** 2)
    assert abs(fn.volume_constraint(np.array([r, h]), 10.0)) < 1e-10


def test_height_constraint_sign():
    assert fn.height_constraint(np.array([1.0, 3.0]), h_max=5.0) < 0
    assert fn.height_constraint(np.array([1.0, 6.0]), h_max=5.0) > 0
