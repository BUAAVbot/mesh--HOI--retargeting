"""Numeric geometry for explicit robot contact adaptation; no dynamics or state writes."""
import numpy as np


def halfspaces(polygon):
    p=np.asarray(polygon,float);e=np.roll(p,-1,axis=0)-p;n=np.c_[-e[:,1],e[:,0]]
    n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
    if np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))<0:n=-n
    return n,np.sum(p*n,axis=1)


def nearest_feasible(point,A,b):
    """Exact projection onto a closed 2-D halfspace intersection, or None."""
    point=np.asarray(point,float);A=np.asarray(A,float).reshape(-1,2);b=np.asarray(b,float)
    if not len(A):return point.copy()
    candidates=[point]
    for n,v in zip(A,b):candidates.append(point+(v-n@point)*n/(n@n))
    for i in range(len(A)):
        for j in range(i):
            pair=A[[i,j]]
            if abs(np.linalg.det(pair))>1e-10:candidates.append(np.linalg.solve(pair,b[[i,j]]))
    c=np.asarray(candidates);good=np.all(c@A.T>=b-2e-7,axis=1)
    if not good.any():return None
    c=c[good];return c[np.argmin(np.sum((c-point)**2,axis=1))]


def rotate_xy(points,yaw):
    c,s=np.cos(yaw),np.sin(yaw);return np.asarray(points)@np.array([[c,s],[-s,c]])


def convex_hull(points):
    pts=sorted(set(map(tuple,np.round(np.asarray(points,float),12))))
    def cross(o,a,b):return(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lo=[];hi=[]
    for p in pts:
        while len(lo)>=2 and cross(lo[-2],lo[-1],p)<=1e-12:lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi)>=2 and cross(hi[-2],hi[-1],p)<=1e-12:hi.pop()
        hi.append(p)
    return np.asarray(lo[:-1]+hi[:-1])


def place_foot(point,yaw,footprint,support_name,spec,margin=.008):
    """Fit the full sole inside its patch and outside all higher solid patches."""
    rel=rotate_xy(footprint,yaw);item=next((v for v in spec['supports']if v['name']==support_name),None)
    if item:
        A,b=halfspaces(item['xy_polygon_m']);b=b+margin-np.min(rel@A.T,axis=0)
        height=item['mode_height_m']
    else:A=np.empty((0,2));b=np.empty(0);height=0.
    center=nearest_feasible(point,A,b)
    if center is None:raise ValueError('Full sole does not fit support '+support_name)
    footpoly=convex_hull(rel);foot_axes=halfspaces(footpoly)[0]
    for upper in sorted(spec['supports'],key=lambda x:x['mode_height_m']):
        if upper['mode_height_m']<=height+.03:continue
        poly=np.asarray(upper['xy_polygon_m']);axes=np.r_[halfspaces(poly)[0],foot_axes]
        pmin=(poly@axes.T).min(0);pmax=(poly@axes.T).max(0);rmin=(rel@axes.T).min(0);rmax=(rel@axes.T).max(0)
        # SAT: one separating axis is enough to prove no intersection.
        projection=center@axes.T
        low=projection+rmax<=pmin-margin;high=projection+rmin>=pmax+margin
        if np.any(low|high):
            choices=[]
            for j in range(len(axes)):
                if low[j]:choices.append((pmin[j]-margin-rmax[j]-projection[j],-axes[j],-pmin[j]+margin+rmax[j]))
                if high[j]:choices.append((projection[j]+rmin[j]-pmax[j]-margin,axes[j],pmax[j]+margin-rmin[j]))
            _,normal,lower=max(choices,key=lambda x:x[0]);A=np.r_[A,normal[None]];b=np.r_[b,lower]
            continue
        candidates=[]
        for n,lo,hi,rlo,rhi in zip(axes,pmin,pmax,rmin,rmax):
            for extra,lower in ((n,hi+margin-rlo),(-n,-lo+margin+rhi)):
                aa=np.r_[A,extra[None]];bb=np.r_[b,lower];candidate=nearest_feasible(point,aa,bb)
                if candidate is not None:candidates.append((float(np.sum((candidate-point)**2)),candidate,aa,bb))
        if not candidates:raise ValueError('Sole cannot clear higher solid '+upper['name']+' while on '+support_name)
        _,center,A,b=min(candidates,key=lambda x:x[0])
    beta=np.asarray(item['top_z_eq_ax_by_c']if item else spec['ground']['z_eq_ax_by_c'])
    return center,float(center@beta[:2]+beta[2])


def segments(expected,names,side):
    n=len(expected);start=0
    while start<n:
        end=start+1
        while end<n and expected[end,side]==expected[start,side]and names[end,side]==names[start,side]:end+=1
        yield start,end,bool(expected[start,side]),str(names[start,side]);start=end


def no_flight_adaptation(expected,names,confidence,sole,vsole,speed,ankle,time,surface,spec):
    """Bridge reconstruction-induced flight gaps with an explicitly adapted support foot."""
    before=expected.copy();before_names=names.copy();before_surface=surface.copy();adapted=np.zeros_like(expected)
    unsupported=~before.any(1);edges=np.diff(np.r_[0,unsupported.astype(int),0]);starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    frame_dt=float(np.median(np.diff(time)));longest=0.;decisions=[]
    def beta(name):
        item=next((s for s in spec['supports']if s['name']==name),None)
        return np.asarray(item['top_z_eq_ax_by_c']if item else spec['ground']['z_eq_ax_by_c'])
    def apply(lo,hi,k,name):
        if hi<=lo:return
        b=beta(name);fresh=~expected[lo:hi,k]
        expected[lo:hi,k]=True;confidence[lo:hi,k]=np.where(fresh,.55,confidence[lo:hi,k]);adapted[lo:hi,k]|=fresh
        # A continued contact stays on its existing physical surface even when the raw fit jitters upward.
        names[lo:hi,k]=name;surface[lo:hi,k]=ankle[lo:hi,k,:2]@b[:2]+b[2]
    for lo,hi in zip(starts,ends):
        duration=float((time[hi]if hi<len(time)else time[-1]+frame_dt)-time[lo]);longest=max(longest,duration)
        previous=expected[lo-1].copy()if lo else np.zeros(2,bool);sample_end=min(hi,lo+max(3,int(round(.12/frame_dt))))
        choices=[]
        for k in range(2):
            if previous[k]:name=str(names[lo-1,k])
            else:
                labels,counts=np.unique(names[lo:sample_end,k],return_counts=True);name=str(labels[np.argmax(counts)])
            b=beta(name);gap=np.median(abs(sole[lo:sample_end,k]-(ankle[lo:sample_end,k,:2]@b[:2]+b[2])))
            score=float(gap+.08*np.median(speed[lo:sample_end,k])+.05*np.median(abs(vsole[lo:sample_end,k]))-.08*previous[k])
            # If the same foot must land at a different level/location next, let the other foot support its swing.
            if hi<len(time)and before[hi,k]and not before[hi,1-k]:
                last=np.flatnonzero(before[:lo,k])
                if len(last)and(name!=str(before_names[hi,k])or np.linalg.norm(ankle[hi,k,:2]-ankle[last[-1],k,:2])>.18):score+=.40
            choices.append((score,k,name))
        _,side,name=min(choices,key=lambda x:x[0]);apply(lo,hi,side,name);handoff=None
        # Keep a short overlap when transferring load; these are inferred robot contacts, not observations.
        prior=np.flatnonzero(previous)
        if not previous[side]and len(prior):
            prior_side=int(prior[np.argmin([speed[lo-1,k]for k in prior])]);apply(lo,min(hi,lo+2),prior_side,str(names[lo-1,prior_side]))
        if hi-lo>=6 and hi<len(time)and before[hi,1-side]:
            incoming=1-side;destination=str(before_names[hi,incoming]);b=beta(destination)
            candidates=np.arange(lo+2,hi-1);gap=abs(sole[candidates,incoming]-(ankle[candidates,incoming,:2]@b[:2]+b[2]));distance=np.linalg.norm(ankle[candidates,incoming,:2]-ankle[hi,incoming,:2],axis=1)
            ready=candidates[(gap<.10)&(distance<.35)]
            # A robot may plant a little earlier than the noisy reconstruction indicates.
            # Leaving the previous foot until the end would make its following swing teleport.
            handoff=int(ready[0])if len(ready)else int(lo+np.ceil(.65*(hi-lo)))
            # Reserve at least250ms of source-clock motion for the outgoing foot's next swing.
            # This is an explicit robot timing adaptation; source frames themselves are untouched.
            handoff=min(handoff,hi-max(4,int(np.ceil(.25/frame_dt))))
            handoff=int(np.clip(handoff,lo+2,hi-2));release=min(hi,handoff+2)
            expected[release:hi,side]=before[release:hi,side];confidence[release:hi,side]=.25;adapted[release:hi,side]=False
            names[release:hi,side]=before_names[release:hi,side];surface[release:hi,side]=before_surface[release:hi,side]
            apply(handoff,hi,incoming,destination)
        if handoff is None and hi<len(time)and before[hi,1-side]and not before[hi,side]:apply(hi,min(len(time),hi+2),side,name)
        decisions.append({'start_frame':int(lo),'end_frame_exclusive':int(hi),'source_duration_s':duration,'adapted_support_side':int(side),'support_surface':name,'handoff_to_other_foot_frame':handoff,'confidence':.55})
    if not expected.any(1).all():raise AssertionError('Walking contact adaptation left a flight frame')
    return before,before_names,adapted,{'old_no_support_frame_count':int(unsupported.sum()),'old_longest_no_support_source_s':longest,'adapted_no_support_frame_count':int((~expected.any(1)).sum()),'no_flight_contact_frames_added_per_side':adapted.sum(0).tolist(),'no_flight_decisions':decisions,'no_flight_contact_confidence':.55,'no_flight_assumption':'Source action is walking/stepping/sitting, without a flight jump. Added contacts are robot gait adaptations, not observed force labels.'}


def turn_step_adaptation(target,yaw,expected,names,confidence,time,body_yaw,footprints,sole_offsets,spec):
    """Adapt the initial floor turn by lifting one foot while the other stays planted."""
    before_yaw=yaw.copy();before_contacts=expected.copy();turn_lift=np.zeros_like(expected);records=[]
    ground=spec['ground']['name'];dt=float(np.median(np.diff(time)));step_frames=max(4,int(round(.267/dt)));cut=int(np.searchsorted(time,3.2,side='left'))
    def wrap(v):return np.arctan2(np.sin(v),np.cos(v))
    def anchor(side,xy,angle):
        center,z=place_foot(xy,angle,footprints[side],ground,spec)
        return np.r_[center,z+sole_offsets[side]-.0002]
    def hold_other(side,start,landing):
        other=1-side;begin=start
        if expected[start,other]and names[start,other]==ground:i=start
        else:
            while begin>0 and not expected[begin-1,other]and names[begin-1,other]==ground:begin-=1
            previous=np.flatnonzero(expected[:begin,other]&(names[:begin,other]==ground))
            if len(previous):i=int(previous[-1])
            else:
                future=np.flatnonzero(expected[start:,other]&(names[start:,other]==ground));i=start+int(future[0])if len(future)else start
        angle=float(yaw[i,other]);position=anchor(other,target[i,other,:2],angle)
        end=landing+1
        if end<cut and not expected[end,other]:
            next_contact=end
            while next_contact<len(time)and not expected[next_contact,other]:next_contact+=1
            if next_contact<cut and names[next_contact,other]==ground:end=next_contact
        while end<len(time)and expected[end,other]and names[end,other]==ground:end+=1
        fresh=~expected[begin:end,other];expected[begin:end,other]=True;names[begin:end,other]=ground;confidence[begin:end,other]=np.where(fresh,.55,confidence[begin:end,other]);target[begin:end,other]=position;yaw[begin:end,other]=angle
        return other,begin,end
    busy_until=0;f=0
    while f<cut:
        if f<busy_until:f+=1;continue
        reused=None
        # Reuse a real planned swing whenever it can accomplish the required redirection.
        for side in range(2):
            if expected[f,side]or names[f,side]!=ground:continue
            landing=f
            while landing<len(time)and not expected[landing,side]:landing+=1
            prior=np.flatnonzero(expected[:f,side]&(names[:f,side]==ground))
            if landing>=cut or names[landing,side]!=ground or not len(prior):continue
            old=float(yaw[int(prior[-1]),side])
            if abs(wrap(body_yaw[landing]-old))>.60:reused=(side,f,landing,old);break
        if reused is not None:side,start,landing,old=reused;kind='reuse_existing_swing'
        else:
            candidates=[side for side in range(2)if expected[f,side]and names[f,side]==ground and abs(wrap(body_yaw[f]-yaw[f,side]))>.60]
            if not candidates or f+step_frames>=cut:f+=1;continue
            direction=float(np.mean([wrap(body_yaw[f]-yaw[f,side])for side in candidates]));preferred=1 if direction>0 else 0
            side=preferred if preferred in candidates else max(candidates,key=lambda k:abs(wrap(body_yaw[f]-yaw[f,k])))
            start=f;landing=f+step_frames;old=float(yaw[max(0,start-1),side]);kind='added_in_place_lift'
        support,restore_begin,restore_end=hold_other(side,start,landing)
        # A full planted phase keeps the new heading; it is released only by a later swing.
        desired=old+float(wrap(body_yaw[landing]-old));end=landing+1
        while end<len(time)and expected[end,side]and names[end,side]==ground:end+=1
        old_position=target[max(0,start-1),side].copy()
        landing_xy=old_position[:2]if kind=='added_in_place_lift'else target[landing,side,:2]
        position=anchor(side,landing_xy,desired)
        expected[start:landing,side]=False;confidence[start:landing,side]=.25;names[start:landing,side]=ground
        target[landing:end,side]=position;yaw[landing:end,side]=desired;expected[landing:end,side]=True;confidence[landing:end,side]=np.maximum(confidence[landing:end,side],.55)
        if kind=='added_in_place_lift':turn_lift[start:landing,side]=True
        records.append({'kind':kind,'side':int(side),'start_frame':int(start),'landing_frame':int(landing),'source_duration_s':float(time[landing]-time[start]),'fixed_support_side':int(support),'support_restored_from_frame':int(restore_begin),'support_fixed_until_frame_exclusive':int(restore_end),'old_foot_yaw_rad':old,'landing_foot_yaw_rad':desired,'source_body_yaw_at_landing_rad':float(body_yaw[landing]),'landing_anchor_xy_shift_m':float(np.linalg.norm(position[:2]-old_position[:2])),'confidence':.55})
        busy_until=landing+2;f=landing
    if not expected.any(1).all():raise AssertionError('Floor turn adaptation created a flight phase')
    # Reconnect all swings to their possibly updated stance anchors, without rotating a planted foot.
    for side in range(2):
        cuts=np.flatnonzero(np.r_[True,np.diff(expected[:,side].astype(int))!=0,True])
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            if expected[lo,side]:continue
            left=max(0,lo-1);right=min(len(time)-1,hi);p0=target[left,side].copy();p1=target[right,side].copy();y0=float(yaw[left,side]);dy=float(wrap(yaw[right,side]-y0));duration=max(time[right]-time[left],1e-9)
            arc=.08 if np.all(turn_lift[lo:hi,side])else .10
            for k in range(lo,hi):
                u=float((time[k]-time[left])/duration);w=u**3*(10-15*u+6*u*u);target[k,side]=(1-w)*p0+w*p1;target[k,side,2]+=4*u*(1-u)*(arc+.5*abs(p1[2]-p0[2]));yaw[k,side]=y0+w*dy
    return {'turn_step_records':records,'turn_step_domain':'Only ground step scheduling before source3.2s; a landed anchor remains fixed until its next swing','turn_heading_threshold_rad':.60,'turn_added_lift_source_frames':step_frames,'turn_source_body_heading_changed':False,'turn_no_support_frame_count':int((~expected.any(1)).sum()),'turn_yaw_override_mask':(abs(wrap(yaw-before_yaw))>1e-8).tolist(),'turn_lift_mask':turn_lift.tolist(),'pre_turn_expected':before_contacts.tolist()}


def build_plan(joints,sole,vsole,time,surface,names,legacy_expected,footprints,sole_offsets,spec,body_yaw=None):
    """Infer credible contacts, lock stance, and join anchors with clear swing arcs."""
    ankle=joints[:,[7,8]];speed=np.linalg.norm(np.gradient(ankle[:,:,:2],time,axis=0),axis=-1)
    toes=joints[:,[10,11],:2]-ankle[:,:,:2]
    yaw=np.unwrap(np.arctan2(toes[:,:,1],toes[:,:,0]),axis=0);source_toe_yaw=yaw.copy()
    gap=abs(sole-surface)
    expected=(gap<.045)&(abs(vsole)<.20)&(speed<.22);confidence=np.where(expected,.8,.25)
    for f in range(len(time)):
        if not expected[f].any():
            eligible=(gap[f]<.08)&(abs(vsole[f])<.3)&(speed[f]<.35)
            if eligible.any():
                k=int(np.argmin(np.where(eligible,speed[f]+2*gap[f],np.inf)));expected[f,k]=True;confidence[f,k]=.45
    # During the final settled sit, explicitly adapt both robot feet to fixed floor support.
    settled=(time[:,None]>=10.4)&(names==spec['ground']['name'])&(gap<.09)
    expected[settled]=True;confidence[settled]=.65
    # Bridge tracking jitter only when the same physical anchor can explain both ends.
    for k in range(2):
        for lo,hi,on,name in list(segments(expected,names,k)):
            if not on and lo>0 and hi<len(time) and hi-lo<=4 and expected[lo-1,k]and expected[hi,k]and names[lo-1,k]==names[hi,k]and np.linalg.norm(ankle[hi,k,:2]-ankle[lo-1,k,:2])<.07 and abs(yaw[hi,k]-yaw[lo-1,k])<.20:
                expected[lo:hi,k]=True;names[lo:hi,k]=names[hi,k];confidence[lo:hi,k]=.55
        for lo,hi,on,name in list(segments(expected,names,k)):
            if on and hi-lo<3 and lo>0 and hi<len(time):expected[lo:hi,k]=False;confidence[lo:hi,k]=.25
    anchor_mask,anchor_names,added_contact,no_flight_audit=no_flight_adaptation(expected,names,confidence,sole,vsole,speed,ankle,time,surface,spec)
    target=ankle.copy();target[:,:,:2]=joints[:,0,None,:2]+.90*(ankle[:,:,:2]-joints[:,0,None,:2])
    target[:,:,2]=np.maximum(sole,surface+.015)+np.asarray(sole_offsets)[None]
    for k in range(2):
        for lo,hi,on,name in list(segments(expected,names,k)):
            if not on:continue
            # Extending a contact must not drag its anchor towards raw moving-foot estimates.
            credible=np.flatnonzero(anchor_mask[lo:hi,k]&(anchor_names[lo:hi,k]==name))+lo
            if len(credible):angle=float(np.median(yaw[credible,k]));center=np.median(target[credible,k,:2],axis=0)
            else:
                item=next((s for s in spec['supports']if s['name']==name),None);b=np.asarray(item['top_z_eq_ax_by_c']if item else spec['ground']['z_eq_ax_by_c'])
                score=abs(sole[lo:hi,k]-(ankle[lo:hi,k,:2]@b[:2]+b[2]))+.08*speed[lo:hi,k];best=lo+int(np.argmin(score));angle=float(yaw[best,k]);center=target[best,k,:2].copy()
            xy,z=place_foot(center,angle,footprints[k],name,spec)
            target[lo:hi,k]=np.r_[xy,z+sole_offsets[k]-.0002];yaw[lo:hi,k]=angle;surface[lo:hi,k]=z
        # Every non-contact interval is joined to its neighbouring locked contacts.
        on_ids=np.flatnonzero(expected[:,k]);cuts=np.flatnonzero(np.r_[True,np.diff(expected[:,k].astype(int))!=0,True])
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            if expected[lo,k]:continue
            left=max(0,lo-1);right=min(len(time)-1,hi)
            p0=target[left,k].copy();p1=target[right,k].copy();y0=yaw[left,k];y1=yaw[right,k]
            duration=max(time[right]-time[left],1e-9)
            for f in range(lo,hi):
                u=float((time[f]-time[left])/duration);w=u**3*(10-15*u+6*u*u)
                target[f,k]=(1-w)*p0+w*p1
                target[f,k,2]+=4*u*(1-u)*(.10+.5*abs(p1[2]-p0[2]))
                yaw[f,k]=y0+w*(y1-y0)
    if body_yaw is None:
        lateral=joints[:,1]-joints[:,2];front=np.cross(lateral,[0,0,1]);body_yaw=np.unwrap(np.arctan2(front[:,1],front[:,0]))
    turn_audit=turn_step_adaptation(target,yaw,expected,names,confidence,time,np.asarray(body_yaw),footprints,sole_offsets,spec)
    target_speed=np.linalg.norm(np.gradient(target[:,:,:2],time,axis=0),axis=-1)
    # Finite-difference speed includes transitions; drift within each locked interval is exactly zero.
    drift=[];yaw_drift=[]
    for k in range(2):
        for lo,hi,on,name in segments(expected,names,k):
            if on:
                drift.append(float(np.linalg.norm(target[lo:hi,k,:2]-target[lo,k,:2],axis=1).max()));yaw_drift.append(float(np.max(abs(np.arctan2(np.sin(yaw[lo:hi,k]-yaw[lo,k]),np.cos(yaw[lo:hi,k]-yaw[lo,k]))))))
    if max(drift or[0.])>1e-7 or max(yaw_drift or[0.])>1e-7:raise AssertionError('Robot turn changed a planted foot anchor')
    audit={'legacy_expected_count':legacy_expected.sum(0).tolist(),'adapted_expected_count':expected.sum(0).tolist(),'contact_flag_changed_count':np.sum(expected!=legacy_expected,axis=0).tolist(),'source_stance_ankle_speed_over_0_3m_s_count':int(np.sum(legacy_expected&(speed>.3))),'source_first_90_expected_count':legacy_expected[:90].sum(0).tolist(),'adapted_first_90_expected_count':expected[:90].sum(0).tolist(),'maximum_locked_stance_xy_drift_m':max(drift or[0.]),'peak_target_xy_speed_source_m_s':target_speed.max(0).tolist(),'source_toe_direction':'joints10-7 and11-8 projected into worldXY; Vita sole +X is its longer forefoot axis','inference':'Robot-adapted contact plan; no claim of observed contact forces','pre_no_flight_expected':anchor_mask.tolist(),'no_flight_adapted_mask':added_contact.tolist()}
    audit.update(no_flight_audit);audit.update(turn_audit);audit['source_toe_yaw_before_robot_adaptation_rad']=source_toe_yaw.tolist();audit['maximum_locked_stance_yaw_drift_rad']=max(yaw_drift or[0.])
    return target,yaw,expected,confidence,audit
