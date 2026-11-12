import unittest

from common import classify_robot_static_bodies


class _Object:
    mjOBJ_BODY = 1


class _Mujoco:
    mjtObj = _Object
    names = ("world", "terrain", "robot/base_link", "robot/left_foot", "seat_0")

    @classmethod
    def mj_id2name(cls, model, kind, body_id):
        return cls.names[body_id]


class _Model:
    nbody = 5
    body_rootid = (0, 1, 2, 2, 4)


class BodyClassificationTest(unittest.TestCase):
    def test_multiple_static_roots_are_not_robot_bodies(self):
        robot, static, order = classify_robot_static_bodies(_Model(), _Mujoco())
        self.assertEqual(robot, {2, 3})
        self.assertEqual(static, {0, 1, 4})
        self.assertEqual(order, [2, 3])


if __name__ == "__main__":
    unittest.main()
