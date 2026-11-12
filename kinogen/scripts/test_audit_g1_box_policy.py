"""Synthetic UNIT fixtures only: never simulation/training/evaluation evidence.

All dummy checkpoints, arrays and audit JSONs live in deleted temporary folders.
Run on an existing CPU NumPy environment; no Isaac, torch, model or GPU is used.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from audit_g1_box_policy import audit_episodes,physics_checks,Checks,sha


class AuditBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='SYNTHETIC-BOX-AUDIT-UNIT-');self.addCleanup(self.temp.cleanup)
        self.dir=Path(self.temp.name);self.checkpoint=self.dir/'NOT_A_POLICY.bin';self.checkpoint.write_bytes(b'SYNTHETIC UNIT FIXTURE ONLY')
        self.checkpoint_sha=sha(self.checkpoint);self.trace=self.dir/'SYNTHETIC_TRACE.npz';self.meta=self.dir/'SYNTHETIC_METADATA.json';self.output=self.dir/'TEST_ONLY_REPORT.json'
    def fixture(self,episodes=1):
        t=np.tile(np.arange(200)*.02,episodes);n=len(t);hold=t>=1
        d={k:np.zeros((n,79))for k in ('joint_pos','joint_vel','joint_targets')}
        d.update(timestamps=t,episode_id=np.repeat(np.arange(episodes),200),seed_id=np.repeat(np.arange(episodes),200),done=np.tile(np.arange(200)==199,episodes),reset_events=np.tile(np.arange(200)==0,episodes),root_pos=np.tile([0.,0.,.8],(n,1)),root_quat=np.tile([1.,0.,0.,0.],(n,1)),box_pos=np.tile([.4,0.,.87],(n,1)),box_quat=np.tile([1.,0.,0.,0.],(n,1)),box_vel=np.zeros((n,6)),hand_contact_force_left_N=hold.astype(float)*2,hand_contact_force_right_N=hold.astype(float)*2,table_contact_force_N=(~hold).astype(float)*9.81,feet_force=np.tile([[0,0,180],[0,0,180]],(n,1,1)),source_checkpoint_sha256=np.asarray(self.checkpoint_sha))
        d['root_quats']=d['root_quat'];d['box_pos'][hold,2]=1.02
        raw={k:{'active_calls':0,'reset_calls':1}for k in ('robot.write_root_state_to_sim','box.write_root_state_to_sim','robot.write_joint_state_to_sim','robot.set_external_force_and_torque','box.set_external_force_and_torque')}
        p={'sources':{'trace_sha256':'unit_fixture','checkpoint_sha256':self.checkpoint_sha},'stage':{'gravity_m_s2':[0,0,-9.81],'box':{'mass_kg':1.,'size_xyz_m':[.28,.34,.24],'dynamic':True,'kinematic':False,'gravity_enabled':True},'table':{'top_z_m':.75},'robot':{'free_base':True,'root_kinematic':False},'object_attachment_joint_count':0,'artificial_grasp_assist':False},'contacts':{'hand_box_filtered':True,'table_box_filtered':True},'trace_layout':{'quaternion_order':'wxyz'},'guards':{'active_state_write_calls':0,'active_external_wrench_calls':0,'started_before_active':True,'instrumented_apis':list(raw),'per_api':raw},'evaluation':{'seeds':list(range(episodes)),'sample_phase':'post_physics_pre_reset','policy_inference_calls':1,'actor_sha256':'b'*64},'training':{'optimizer_steps':1,'environment_steps':1,'initial_actor_sha256':'a'*64,'final_actor_sha256':'b'*64},'SYNTHETIC_UNIT_FIXTURE_NOT_PHYSICS':True}
        return d,p
    def episode(self,d,p):
        from types import SimpleNamespace
        args=SimpleNamespace(checkpoint=self.checkpoint,episodes=1,hand_force=.5,table_force=.2,lift_height=.1,hold_seconds=2.,min_root_height=.45,max_root_tilt_deg=35.,min_feet_force=5.)
        return audit_episodes(d,p,args,Checks())[0][0]
    def test_terminal_hold_pregrasp_reset_and_bottom_shortcuts(self):
        d,p=self.fixture();self.assertTrue(self.episode(d,p)['success'])
        for failure in ('early_hold','pregrasp','active_reset','continued_after_done','com_rotation','fall','not_table_rest'):
            with self.subTest(failure=failure):
                x=copy.deepcopy(d)
                if failure=='early_hold':x['hand_contact_force_left_N'][-10:]=0
                elif failure=='pregrasp':x['hand_contact_force_left_N'][0]=2
                elif failure=='active_reset':x['reset_events'][100]=True
                elif failure=='continued_after_done':x['done'][100]=True
                elif failure=='com_rotation':
                    hold=x['timestamps']>=1;x['box_pos'][hold,2]=.97;x['box_quat'][hold]=[2**-.5,2**-.5,0,0]
                elif failure=='fall':x['root_pos'][-1,2]=.2
                else:x['table_contact_force_N'][0]=0
                self.assertFalse(self.episode(x,p)['success'])
    def test_guard_raw_counters_and_real_actor_update_required(self):
        _,p=self.fixture();self.assertTrue(physics_checks(p,'unit_fixture',self.checkpoint_sha,Checks()))
        for failure in ('missing_guard','false_zero_aggregate','unchanged_actor'):
            with self.subTest(failure=failure):
                x=copy.deepcopy(p)
                if failure=='missing_guard':del x['guards']['active_state_write_calls']
                elif failure=='false_zero_aggregate':x['guards']['per_api']['box.write_root_state_to_sim']['active_calls']=1
                else:x['training']['initial_actor_sha256']='b'*64
                self.assertFalse(physics_checks(x,'unit_fixture',self.checkpoint_sha,Checks()))
    def test_cli_trace_only_never_certifies_delivery_and_cannot_overwrite_source(self):
        d,p=self.fixture(32);np.savez_compressed(self.trace,**d);p['sources']['trace_sha256']=sha(self.trace);self.meta.write_text(json.dumps(p))
        cmd=[sys.executable,str(Path(__file__).with_name('audit_g1_box_policy.py')),'--trace',str(self.trace),'--physics',str(self.meta),'--checkpoint',str(self.checkpoint),'--trace-only','--output',str(self.output)]
        result=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        report=json.loads(self.output.read_text());self.assertEqual(report['successful_episodes'],32);self.assertFalse(report['final_delivery_verified']);self.assertEqual(report['status'],'trace_feedback_only')
        original=sha(self.trace);cmd[-1]=str(self.trace);result=subprocess.run(cmd,capture_output=True,text=True)
        self.assertEqual(result.returncode,2);self.assertEqual(sha(self.trace),original)


if __name__=='__main__':
    print('SYNTHETIC UNIT TESTS ONLY. No simulator, trained policy, or actual task result is evaluated.',flush=True)
    unittest.main(verbosity=2)
