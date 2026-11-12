"""Regression: a second scene object's contacts and penetration must be active."""
from pathlib import Path
import sys
import unittest
import tempfile
from types import SimpleNamespace
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import retarget_smpl_to_humanoid_surface_vector as engine
from object_scene_sources import combine_object_sources


class ObjectSceneTest(unittest.TestCase):
    def test_floating_xml_stays_with_result(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "assets/robot.xml"
            source.parent.mkdir()
            text = ('<mujoco><worldbody><body name="robot"><freejoint/>'
                    '<geom type="sphere" size=".1"/></body></worldbody></mujoco>')
            source.write_text(text)
            output = directory / "output/motion.npz"
            args = SimpleNamespace(
                robot_xml=str(source), out=str(output), force_no_floating_root=False,
                config_data={"robot": {"name": "robot", "xml": str(source),
                                       "point_cloud_center": "body:robot"}},
            )
            generated = engine.prepare_robot_xml(args)
            self.assertEqual(generated, output.with_suffix(".floating_mjcf.xml"))
            self.assertEqual(source.read_text(), text)
            self.assertEqual(list(source.parent.iterdir()), [source])
            self.assertEqual(mujoco.MjModel.from_xml_path(str(generated)).nq, 7)

    def test_nearest_contacts_cover_both_objects(self):
        def source(name, distances, point):
            return dict(name=name, path=name, distances=np.array([distances]),
                        object_ids=np.zeros((1,2),dtype=int),
                        pair_vectors=np.array([[[point,0,0],[point,0,0]]]),
                        source_points_world=np.array([[[point,0,0]]]),
                        retarget_points_world=np.array([[[point,0,0]]]),
                        motion_positions=np.zeros((1,3)), motion_quats_wxyz=np.array([[1,0,0,0]]))
        held=source('held',[.01,.5],1.)
        table=source('table',[.5,.01],2.)
        scene=combine_object_sources([held,table])
        np.testing.assert_array_equal(scene['object_ids'],[[0,1]])
        np.testing.assert_array_equal(scene['pair_vectors'][0,:,0],[1.,2.])
        self.assertEqual(scene['motion_positions'].shape,(1,2,3))
        self.assertIs(combine_object_sources([held]),held)

    def test_only_second_object_intersects_robot(self):
        xml='<mujoco><worldbody><body name="robot"><freejoint/><geom name="robot_sphere" type="sphere" size="0.1"/></body><body name="object"><freejoint/><geom name="object_sphere" type="sphere" size="0.2"/></body></worldbody></mujoco>'
        def cache():
            model=mujoco.MjModel.from_xml_string(xml)
            return dict(model=model,data=mujoco.MjData(model),robot_nq=7,robot_nv=6,object_qadr=7,
                        robot_geom_ids=np.array([0]),object_geom_ids=np.array([1]),
                        labels=engine.common.geom_collision_labels(model))
        qpos=np.array([0,0,0,1,0,0,0.])
        far=np.array([2,0,0,1,0,0,0.])
        near=np.array([.15,0,0,1,0,0,0.])
        _,far_dist=engine.compute_robot_object_penetration_rows(qpos,far,cache(),0,.5,128)
        expected_rows,expected_dist=engine.compute_robot_object_penetration_rows(qpos,near,cache(),0,.5,128)
        rows,dist=engine.compute_robot_object_penetration_rows(qpos,np.stack([far,near]),{'objects':[cache(),cache()]},0,.5,128)
        self.assertFalse(far_dist)
        self.assertTrue(len(dist))
        np.testing.assert_allclose(rows,expected_rows)
        np.testing.assert_allclose(dist,expected_dist)
        self.assertLess(min(dist),-.14)


if __name__=='__main__':
    unittest.main()
