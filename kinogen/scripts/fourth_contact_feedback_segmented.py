"""Versioned finger hook for independent proximal/distal hand collision meshes.

The original frozen feedback and table-guard modules remain unchanged. Distance
goals use real distal collision meshes; force sensing and table clearance retain
all segments, including proximal links and palms. The API is unchanged.
"""
import numpy as np
import mujoco as mj
from fourth_contact_feedback_tableguard import ReferenceContactFeedbackWithTableGuard
from fourth_contact_feedback import _read_config
from fourth_convex_separation import ConvexSeparationOracle


class SegmentedReferenceContactFeedback(ReferenceContactFeedbackWithTableGuard):
    def __init__(self,model,reference,config=None):
        config=_read_config(config)
        backend=config.pop('geometry_backend','convex_sat')
        if backend!='convex_sat':raise ValueError('Segmented feedback requires geometry_backend=convex_sat')
        super().__init__(model,reference,config)
        self._sat=ConvexSeparationOracle(model)
        self.plane_proof_queries=0
        numeric=mj.mj_name2id(model,mj.mjtObj.mjOBJ_NUMERIC,'source_table_plane_abcd')
        if numeric<0:raise ValueError('Segmented guard requires source tabletop plane metadata')
        adr=int(model.numeric_adr[numeric]);plane=model.numeric_data[adr:adr+4].copy()
        plane/=np.linalg.norm(plane[:3]);self._plane=plane
        self._table_plane_proof_valid=True
        mj.mj_kinematics(model,self._guard_kin)
        for g in self._table_geoms:
            vertices=self._world_mesh_vertices(self._guard_kin,g)
            if vertices is None or np.max(vertices@plane[:3]+plane[3])>2e-7:
                self._table_plane_proof_valid=False
        for hand in self.hands:
            if not all(hand['geometry_controlled']):
                raise ValueError('Segmented feedback requires active geometry downstream of all 22 finger joints')
            for digit in hand['digits']:
                digit['all_segment_geoms']=list(digit['tip_geoms'])
                distal=[g for g in digit['tip_geoms']if int(model.geom_bodyid[g])==int(digit['tip_body'])]
                if not distal:raise ValueError('Missing real distal collision for '+hand['prefix']+digit['name'])
                digit['tip_geoms']=distal

    def _world_mesh_vertices(self,kin,g):
        mesh=int(self.model.geom_dataid[g])
        if self.model.geom_type[g]!=mj.mjtGeom.mjGEOM_MESH:return None
        adr=int(self.model.mesh_vertadr[mesh]);count=int(self.model.mesh_vertnum[mesh])
        vertices=np.asarray(self.model.mesh_vert[adr:adr+count],float)
        return vertices@kin.geom_xmat[g].reshape(3,3).T+kin.geom_xpos[g]

    def _plane_lower_bound(self,kin,g):
        normal=self._plane[:3];rotation=kin.geom_xmat[g].reshape(3,3)
        centre=float(normal@kin.geom_xpos[g]+self._plane[3]);local=rotation.T@normal
        kind=self.model.geom_type[g];size=self.model.geom_size[g]
        if kind==mj.mjtGeom.mjGEOM_MESH:
            return float(np.min(self._world_mesh_vertices(kin,g)@normal+self._plane[3]))
        if kind==mj.mjtGeom.mjGEOM_SPHERE:return centre-size[0]
        if kind==mj.mjtGeom.mjGEOM_CAPSULE:return centre-size[0]-size[1]*abs(local[2])
        if kind==mj.mjtGeom.mjGEOM_BOX:return centre-float(abs(local)@size)
        if kind==mj.mjtGeom.mjGEOM_CYLINDER:return centre-size[1]*abs(local[2])-size[0]*np.linalg.norm(local[:2])
        if kind==mj.mjtGeom.mjGEOM_ELLIPSOID:return centre-np.linalg.norm(local*size)
        return -np.inf

    def _clearance(self,data,hand_index,targets):
        # A collider entirely above the top halfspace cannot intersect a table
        # proven entirely below it. The bound uses all compiled mesh vertices,
        # so mesh re-centring/scaling and convexification are accounted for.
        if not self._table_plane_proof_valid:return super()._clearance(data,hand_index,targets)
        hand=self.hands[hand_index];kin=self._guard_kin
        kin.qpos[:]=data.qpos;kin.qvel[:]=data.qvel;kin.qpos[hand['qa']]=targets
        mj.mj_kinematics(self.model,kin)
        margin=float(self.guard_config['minimum_clearance_m']);above=[];remaining=[]
        for g in self._guard_geoms[hand_index]:
            bound=self._plane_lower_bound(kin,g)
            if bound>=margin:above.append((g,bound));self.plane_proof_queries+=1
            else:remaining.append(g)
        original=self._guard_geoms[hand_index]
        try:
            self._guard_geoms[hand_index]=remaining
            minimum,fixed,uncertain=super()._clearance(data,hand_index,targets)
        finally:self._guard_geoms[hand_index]=original
        for g,bound in above:
            minimum=min(minimum,bound)
            if g in self._fixed_geoms[hand_index]:fixed=min(fixed,bound)
        return minimum,fixed,uncertain

    def _geometry(self,hand,object_index,jacobian=False):
        distances,J=super()._geometry(hand,object_index,jacobian)
        # Near contact, EPA distance can jump by millimetres after a microradian
        # perturbation. Use a deterministic separating-axis constraint instead.
        # This is a distinct conservative signed gap, not Euclidean distance.
        def bounds(g,query_superset=False):
            rotation=self._kin.geom_xmat[g].reshape(3,3)
            centre=self._kin.geom_xpos[g]+rotation@self.model.geom_aabb[g,:3]
            half=np.abs(rotation)@self.model.geom_aabb[g,3:]+1e-7
            if query_superset and self.model.geom_type[g]==mj.mjtGeom.mjGEOM_CYLINDER:
                error=float(self.model.geom_size[g,0])*(1/np.cos(np.pi/32)-1)
                half+=error*np.sum(abs(rotation[:,:2]),axis=1)
            return centre-half,centre+half
        boxes=[bounds(g)for g in self.object_geoms[object_index]]
        lower=np.min([x[0]for x in boxes],axis=0);upper=np.max([x[1]for x in boxes],axis=0)
        for row,digit in enumerate(hand['digits']):
            # Backend selection must not depend on an unreliable EPA value.
            # AABB distance is a deterministic lower bound on real separation.
            box_distance=min(float(np.linalg.norm(np.maximum(np.maximum(lower-bounds(g)[1],bounds(g)[0]-upper),0.)))for g in digit['tip_geoms'])
            if box_distance>.025:continue
            candidates=[]
            for tip in digit['tip_geoms']:
                for obj in self.object_geoms[object_index]:
                    la,ua=bounds(tip);lb,ub=bounds(obj,True)
                    lower_bound=float(np.max(np.maximum(la-ub,lb-ua)))
                    candidates.append((lower_bound,tip,obj))
            best=(np.inf,-1,-1)
            for lower_bound,tip,obj in sorted(candidates):
                # World XYZ separation axes are included in SAT. Consequently
                # this conservative AABB-axis gap is a proven lower bound on
                # the pair's SAT gap, so distant pieces can be skipped exactly.
                if lower_bound>best[0]+1e-10:continue
                gap,_=self._sat.query(self._kin,tip,obj)
                if gap<best[0]:best=(gap,tip,obj)
            gap,tip,obj=best
            distances[row]=gap
            if jacobian:
                _,J[row]=self._sat.query(self._kin,tip,obj,hand['va'])
        return distances,J

    def hand_geometry_contract(self):
        rows=super().hand_geometry_contract()
        for row in rows:
            site=mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_SITE,row['hand']+'palm_control_site')
            if site<0:raise ValueError('Missing explicit stable palm control site')
            row['palm_collision_aabb_center_in_hand_base_m']=row['palm_center_site_in_hand_base_m']
            row['palm_center_site_in_hand_base_m']=self.model.site_pos[site].tolist()
            row['explicit_palm_control_site']=self.model.site(site).name
            row['method']='Explicit unchanged palm control site; real distal collision AABB and native point Jacobian.'
        return rows

    def report(self):
        report=super().report();report['schema']='fourth.segmented_contact_feedback.v2'
        report['configuration']['geometry_backend']='convex_sat'
        report['limitations']=[s for s in report['limitations']if not s.startswith('Four recovered whole-finger collision proxies')]
        report['limitations'].append('Distal mesh distance goals do not establish opposed contact or force closure; those require actual physical contact checks.')
        report['segmented_hand_geometry']={
            'all_22_native_finger_joints_geometry_controlled':all(all(h['geometry_controlled'])for h in self.hands),
            'distance_targets':'Actual distal collision meshes only; no fictitious visual tip shape.',
            'force_sensing':'Every actual thumb/finger collision segment; palm included in total hand force.',
            'table_guard':'Every active hand/wrist collision segment, with all 22 finger channels eligible.',
            'hands':[{'prefix':h['prefix'],'digits':[{'digit':d['name'],
                'distal_distance_geoms':[self.model.geom(g).name for g in d['tip_geoms']],
                'all_active_segment_geoms':[self.model.geom(g).name for g in d['all_segment_geoms']]}
                for d in h['digits']]}for h in self.hands]}
        report['table_clearance_guard']['table_halfspace_separation_proof']={
            'table_all_vertices_below_source_plane_verified':self._table_plane_proof_valid,
            'source_plane_abcd':self._plane.tolist(),'successful_early_separation_queries':self.plane_proof_queries,
            'method':'All compiled convex-mesh vertices above source plane by required margin; exact mesh distance remains fallback near table/edges.',
            'reported_clearance_can_be_conservative_plane_lower_bound':True}
        report['segmented_hand_geometry']['near_contact_signed_gap_oracle']=self._sat.report()
        report['segmented_hand_geometry']['legacy_selected_tip_distance_m_semantics']='When distal/object-union AABBs are within25mm: conservative convex SAT separation gap; otherwise validated separated Euclidean GJK distance.'
        report['segmented_hand_geometry']['constraint_history_field']='selected_tip_constraint_gap_m; legacy selected_tip_distance_m is retained as an identical compatibility alias and is not always Euclidean distance.'
        return report

    def history_arrays(self):
        arrays=super().history_arrays()
        if 'selected_tip_distance_m'in arrays:
            arrays['selected_tip_constraint_gap_m']=arrays['selected_tip_distance_m'].copy()
        return arrays
