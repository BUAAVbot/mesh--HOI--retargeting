"""Check that AABB pruning preserves the exact capped native distance."""
import unittest
import mujoco as mj
import numpy as np
from sbto_adapter import geom_bounds,nearby_supports


class SupportBroadphaseTests(unittest.TestCase):
    def test_native_distances_unchanged(self):
        model=mj.MjModel.from_xml_string('''<mujoco><asset><mesh name="prism"
          vertex="0 0 0 1 0 0 0 1 0 0 0 -.2 1 0 -.2 0 1 -.2"/></asset>
          <worldbody><geom type="mesh" mesh="prism" pos="1 0 .2"/>
          <geom type="box" size=".2 .2 .1" pos="-1 0 0"/>
          <body pos="0 0 1"><freejoint/><geom type="box" size=".08 .04 .025" mass="1"/>
          </body></worldbody></mujoco>''')
        data=mj.MjData(model);mj.mj_forward(model,data)
        support=[0,1];feet=[2];bounds=geom_bounds(model,data,support)
        rng=np.random.default_rng(21)
        for i in range(100):
            data.qpos[:3]=rng.uniform([-1.5,-.5,-.1],[2,.8,1.])
            data.qpos[3:7]=rng.normal(size=4);data.qpos[3:7]/=np.linalg.norm(data.qpos[3:7])
            mj.mj_forward(model,data)
            near=nearby_supports(model,data,feet,support,bounds)
            brute=min(mj.mj_geomDistance(model,data,2,g,.2,None) for g in support)
            pruned=min((mj.mj_geomDistance(model,data,2,int(g),.2,None) for g in near),default=.2)
            self.assertAlmostEqual(brute,pruned,places=12)


if __name__=='__main__':unittest.main()
