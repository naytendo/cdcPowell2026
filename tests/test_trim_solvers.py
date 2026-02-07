import numpy as np
from warm_start.trim import _central_jacobian_z

class FakeSys:
    """
    A simple fake continuous model where:
        dv     = alpha
        dgamma = thr
        dh     = 0
    So f_ct(x, z, p) = [alpha, thr, 0].
    The Jacobian wrt z = [alpha, thr] is exactly the identity matrix.
    """
    def __init__(self):
        self.params = {}

    def f_ct(self, x, z, p):
        return np.array([z[0], z[1], 0.0], float)


def test_central_jacobian_z_identity():
    sys = FakeSys()
    x = np.array([100.0, 0.0, 1000.0])
    z = np.array([0.1, 0.2])

    J = _central_jacobian_z(sys, x, z)

    # Expected Jacobian is identity
    J_expected = np.eye(2)

    assert np.allclose(J, J_expected, atol=1e-6), \
        f"Jacobian incorrect. Got {J}, expected {J_expected}"


def test_central_jacobian_z_symmetry():
    """
    Central differences should be symmetric:
    J(z + eps) - J(z - eps) should be small.
    """
    sys = FakeSys()
    x = np.array([120.0, 0.05, 1500.0])
    z = np.array([0.05, 0.3])

    J1 = _central_jacobian_z(sys, x, z)
    J2 = _central_jacobian_z(sys, x, z + np.array([1e-6, -1e-6]))

    assert np.allclose(J1, J2, atol=1e-4), \
        f"Central difference symmetry violated. J1={J1}, J2={J2}"
