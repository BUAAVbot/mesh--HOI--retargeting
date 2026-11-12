import unittest
import numpy as np
from shoot_pd_controls import orientation_error


class OrientationTests(unittest.TestCase):
    def test_quaternion_sign_and_known_angle(self):
        angle=.3;target=np.array([[1.,0.,0.,0.]])
        actual=np.array([[np.cos(angle/2),0.,0.,np.sin(angle/2)]])
        np.testing.assert_allclose(orientation_error(actual,target),[[0.,0.,angle]],atol=1e-12)
        np.testing.assert_allclose(orientation_error(-actual,target),[[0.,0.,angle]],atol=1e-12)
        np.testing.assert_allclose(orientation_error(actual,actual),0.,atol=1e-12)


if __name__=='__main__':unittest.main()
