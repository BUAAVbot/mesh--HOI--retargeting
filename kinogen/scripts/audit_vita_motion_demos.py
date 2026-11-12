#!/usr/bin/env python3
"""Independent stdlib-only source/NPZ audit and CPU full-frame ffprobe audit."""
from pathlib import Path
import argparse,ast,collections,hashlib,io,itertools,json,math,pickle,pickletools,re,shutil,struct,subprocess,sys,tarfile,time,zipfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
WORKSPACE_ROOT=next((p for p in ROOT.parents if (p/'manifests/repository_layout.json').is_file()),ROOT.parent)
SOURCE_TAR=WORKSPACE_ROOT/'archives/raw_handoffs/vita17/pkls.tar'
if not SOURCE_TAR.is_file():SOURCE_TAR=WORKSPACE_ROOT/'pkls.tar'
def digest(data):return hashlib.sha256(data).hexdigest()
def sha(path):return digest(Path(path).read_bytes())
def product(shape):return math.prod(shape)
class SafeDtype:
 def __init__(self,code,*args):
  if not isinstance(code,str)or not re.fullmatch(r'(?:[fiu][1248]|b1|O8|U[0-9]+|S[0-9]+)',code):raise ValueError('Unsupported primitive dtype')
  self.code=code;self.endian='<';self.state=None
 def __setstate__(self,state):
  if not isinstance(state,tuple)or len(state)!=8 or state[0]!=3 or state[1]not in ('<','>','|','=')or any(x is not None for x in state[2:5]):raise ValueError('Structured/nonprimitive dtype rejected')
  self.endian=state[1];self.state=state
 @property
 def width(self):return int(self.code[1:])*(4 if self.code[0]=='U'else 1)
class SafeArray:
 def __setstate__(self,state):
  if not isinstance(state,tuple)or len(state)!=5 or state[0]!=1:raise ValueError('Unsupported ndarray state')
  _,shape,dtype,fortran,data=state
  if not isinstance(shape,tuple)or len(shape)>6 or any(type(x)is not int or x<0 for x in shape)or product(shape)>10000000:raise ValueError('Invalid array shape')
  if not isinstance(dtype,SafeDtype)or type(fortran)is not bool:raise ValueError('Invalid ndarray type/order')
  if dtype.code=='O8':
   if len(shape)!=1 or not isinstance(data,list)or len(data)!=product(shape)or not all(isinstance(x,str)for x in data):raise ValueError('Object arrays restricted to primitive joint-name strings')
  elif not isinstance(data,bytes)or len(data)!=product(shape)*dtype.width:raise ValueError('Numeric array byte length mismatch')
  self.shape=shape;self.dtype=dtype;self.fortran=fortran;self.data=data
 def c_bytes(self):
  if self.dtype.code=='O8':raise ValueError('Object array has no numeric byte representation')
  if not self.fortran or len(self.shape)<2:return self.data
  width=self.dtype.width;strides=[product(self.shape[:i])for i in range(len(self.shape))]
  return b''.join(self.data[sum(c*s for c,s in zip(index,strides))*width:(sum(c*s for c,s in zip(index,strides))+1)*width]for index in itertools.product(*(range(x)for x in self.shape)))
 def values(self):
  if self.dtype.code=='O8':return self.data.copy()
  b=self.c_bytes();code=self.dtype.code;kind=code[0];width=self.dtype.width;endian=self.dtype.endian
  if kind=='U':return [b[k:k+width].decode('utf-32-be'if endian=='>'else 'utf-32-le').rstrip('\0')for k in range(0,len(b),width)]
  if kind=='S':return [b[k:k+width].rstrip(b'\0').decode('utf8')for k in range(0,len(b),width)]
  formats={'f4':'f','f8':'d','i1':'b','i2':'h','i4':'i','i8':'q','u1':'B','u2':'H','u4':'I','u8':'Q','b1':'?'}
  if code not in formats:raise ValueError('Unsupported numerical dtype')
  return [x[0]for x in struct.iter_unpack(('>'if endian=='>'else '<')+formats[code],b)]
def reconstruct(cls,shape,typecode):
 if cls is not SafeArray or shape!=(0,)or typecode!=b'b':raise ValueError('Unexpected NumPy reconstruct arguments')
 return SafeArray()
def scalar(dtype,data):
 x=SafeArray();x.__setstate__((1,(),dtype,False,data));return x.values()[0]
class RestrictedReader(pickle.Unpickler):
 def find_class(self,module,name):
  allowed={('numpy','dtype'):SafeDtype,('numpy','ndarray'):SafeArray,('numpy._core.multiarray','_reconstruct'):reconstruct,('numpy.core.multiarray','_reconstruct'):reconstruct,('numpy._core.multiarray','scalar'):scalar,('numpy.core.multiarray','scalar'):scalar}
  if(module,name)not in allowed:raise pickle.UnpicklingError('Rejected global: '+module+'.'+name)
  return allowed[module,name]
 def persistent_load(self,pid):raise pickle.UnpicklingError('Persistent objects rejected')
def source_pickle(raw):
 for op,_,_ in pickletools.genops(raw):
  if op.name in ('PERSID','BINPERSID','EXT1','EXT2','EXT4','INST','OBJ','NEWOBJ','NEWOBJ_EX'):raise ValueError('Rejected pickle construction opcode '+op.name)
 obj=RestrictedReader(io.BytesIO(raw)).load()
 if not isinstance(obj,dict):raise ValueError('Expected a primitive dictionary')
 return obj
def npy(raw):
 if raw[:6]!=b'\x93NUMPY':raise ValueError('Missing NPY magic')
 major,minor=raw[6:8]
 if major==1:length=struct.unpack('<H',raw[8:10])[0];start=10
 elif major in (2,3):length=struct.unpack('<I',raw[8:12])[0];start=12
 else:raise ValueError('Unsupported NPY version')
 if length>65536:raise ValueError('Oversize NPY header')
 h=ast.literal_eval(raw[start:start+length].decode('utf8'if major==3 else 'latin1'));descr=h['descr']
 if not isinstance(descr,str)or descr[0]not in '<>|='or descr[1:].startswith('O'):raise ValueError('Object/structured NPY rejected (allow_pickle=False contract)')
 d=SafeDtype(descr[1:]);d.endian=descr[0];x=SafeArray();x.__setstate__((1,h['shape'],d,h['fortran_order'],raw[start+length:]));return x
def npz(path):
 with zipfile.ZipFile(path)as z:
  names=z.namelist()
  if len(names)!=len(set(names))or any(not n.endswith('.npy')or '/'in n or '\\'in n for n in names):raise ValueError('Unexpected NPZ member')
  if sum(x.file_size for x in z.infolist())>67108864:raise ValueError('NPZ size limit exceeded')
  return {n[:-4]:npy(z.read(n))for n in names}
def finite_summary(values):
 bad=[i for i,v in enumerate(values)if not math.isfinite(v)];return {'finite':not bad,'nonfinite_count':len(bad),'first_nonfinite_flat_indices':bad[:10]}
def quantile(values,p):
 if not values:return None
 x=sorted(values);i=(len(x)-1)*p;a=int(i);b=min(a+1,len(x)-1);return x[a]*(b-i)+x[b]*(i-a)if a!=b else x[a]
def write_json(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8');tmp.replace(path)
def urdf_joints(path):
 answer=[]
 for j in ET.parse(path).getroot().findall('joint'):
  if j.attrib['type']=='fixed':continue
  limit=j.find('limit');answer.append({'name':j.attrib['name'],'type':j.attrib['type'],'limits':{}if limit is None else {k:float(v)for k,v in limit.attrib.items()}})
 return answer
def audit_inputs(args):
 began=time.monotonic();data=args.data_root;manifestpath=data/'input_manifest.json';manifest=json.loads(manifestpath.read_text(encoding='utf8'));joints=urdf_joints(args.urdf);names=[j['name']for j in joints];archive_sha=sha(args.source_tar);errors=[];rows=[];raw_groups=collections.defaultdict(list);motion_groups=collections.defaultdict(list)
 if archive_sha!=manifest['source_motion_tar_sha256']:errors.append('Original source archive SHA differs from conversion manifest')
 with tarfile.open(args.source_tar,'r:*')as archive:
  members={m.name:m for m in archive.getmembers()if m.isfile()and m.name.endswith('.pkl')}
  if len(members)!=17:errors.append('Source archive does not contain exactly17 PKLs')
  for item in manifest['motions']:
   member=item['source_member'];raw=archive.extractfile(members[member]).read();original_sha=digest(raw);raw_groups[original_sha].append(member);src=source_pickle(raw);path=data/item['path'];converted=npz(path);clip_errors=[];source_names=src['dof_names'].values();fps=src['fps'];n=src['root_pos'].shape[0];corechecks={};layout={};arrays={}
   for original,key in [('root_pos','root_pos'),('root_rot','root_rot_xyzw'),('dof_pos','dof_pos')]:
    a=src[original];b=converted[key];equal=a.shape==b.shape and a.dtype.code==b.dtype.code and a.c_bytes()==b.c_bytes();corechecks[key]={'element_bytes_equal_in_logical_C_order':equal,'shape':list(a.shape),'source_dtype':a.dtype.code,'source_fortran_order':a.fortran,'npz_fortran_order':b.fortran,'source_logical_c_sha256':digest(a.c_bytes()),'converted_logical_c_sha256':digest(b.c_bytes())};arrays[key]=a.values()
    if not equal:clip_errors.append(key+' conversion mismatch')
   allnames=source_names==converted['dof_names'].values()==names;sourcebind=original_sha==item['source_pickle_sha256']==converted['source_pickle_sha256'].values()[0];npzbind=sha(path)==item['sha256'];fpsmatch=fps==converted['fps'].values()[0]==item['fps']==30;shapeok=src['root_pos'].shape==(n,3)and src['root_rot'].shape==(n,4)and src['dof_pos'].shape==(n,52)and n==item['frames']and len(names)==52 and len(set(source_names))==52
   for label,ok in [('joint names/order',allnames),('source pickle SHA',sourcebind),('NPZ SHA',npzbind),('fps30',fpsmatch),('tensor shapes',shapeok),('xyzw quaternion convention',src.get('root_rot_format')=='xyzw')]:
    if not ok:clip_errors.append(label+' mismatch')
   fin={k:finite_summary(v)for k,v in arrays.items()}
   if not all(x['finite']for x in fin.values()):clip_errors.append('nonfinite original trajectory')
   q=arrays['root_rot_xyzw'];norms=[math.sqrt(sum(v*v for v in q[i:i+4]))for i in range(0,len(q),4)];normerr=[abs(v-1)for v in norms];qgood=all(math.isfinite(x)and x>1e-12 for x in norms);qmax=max(normerr)if qgood else None
   if not qgood or qmax>.001:clip_errors.append('quaternion norm outside diagnostic replay tolerance.001')
   speeds=[];sign_flips=0
   for i in range(n-1):
    dot=sum(q[i*4+k]*q[(i+1)*4+k]for k in range(4))/(norms[i]*norms[i+1])if qgood else 1.;sign_flips+=dot<0;speeds.append(2*math.acos(min(1,abs(dot)))*fps)
   dof=arrays['dof_pos'];jointrows=[]
   for k,j in enumerate(joints):
    values=dof[k::52];v=[abs(values[i+1]-values[i])*fps for i in range(n-1)];lo=j['limits'].get('lower');hi=j['limits'].get('upper');lim=j['limits'].get('velocity');below=[i for i,x in enumerate(values)if lo is not None and x<lo-1e-6];above=[i for i,x in enumerate(values)if hi is not None and x>hi+1e-6];over=[i for i,x in enumerate(v)if lim is not None and lim>0 and x>lim+1e-6];jointrows.append({'name':j['name'],'minimum_position':min(values),'maximum_position':max(values),'urdf_limits':j['limits'],'position_below_count':len(below),'position_above_count':len(above),'max_position_violation':max([0]+([]if lo is None else[lo-min(values)])+([]if hi is None else[max(values)-hi])),'first_outside_frames':sorted(below+above)[:8],'raw_finite_difference_max_velocity_per_s':max(v,default=0),'raw_finite_difference_p99_velocity_per_s':quantile(v,.99),'velocity_limit_exceeded_intervals':len(over),'first_velocity_exceeded_intervals':over[:8]})
   p=arrays['root_pos'];rootv=[math.sqrt(sum((p[(i+1)*3+k]-p[i*3+k])**2 for k in range(3)))*fps for i in range(n-1)];h=hashlib.sha256()
   for key in ('root_pos','root_rot_xyzw','dof_pos'):h.update(src['root_rot'if key=='root_rot_xyzw'else key].c_bytes())
   h.update(json.dumps(source_names).encode());h.update(str(fps).encode());motion_groups[h.hexdigest()].append(member);selfmeta=src.get('self_collision_refinement');metadata=None if not isinstance(selfmeta,dict)else {k:selfmeta.get(k)for k in ('schema_version','method','initial_collision_frames','final_collision_frames','initial_max_depth_mm','final_max_depth_mm')if isinstance(selfmeta.get(k),(str,int,float,bool,type(None)))}
   rows.append({'clip':path.stem,'source_member':member,'source_pickle_bytes':len(raw),'source_pickle_sha256':original_sha,'npz':{'path':str(path.relative_to(ROOT)),'sha256':sha(path)},'frames':n,'fps':fps,'sample_time_span_seconds':(n-1)/fps,'encoded_video_expected_duration_seconds':n/fps,'joint_names_order_exact':allnames,'source_pickle_binding_exact':sourcebind,'npz_manifest_binding_exact':npzbind,'all_numeric_elements_byte_exact':all(x['element_bytes_equal_in_logical_C_order']for x in corechecks.values()),'numeric_array_checks':corechecks,'finite_checks':fin,'quaternion_order':'xyzw','quaternion_norm_max_error':qmax,'quaternion_norm_mean_error':sum(normerr)/len(normerr),'quaternion_sign_flips':sign_flips,'root_shortest_orientation_max_velocity_rad_s':max(speeds,default=0),'root_translation_max_velocity_m_s':max(rootv,default=0),'joint_limit_and_velocity_diagnostics':jointrows,'source_recorded_self_collision_refinement_unverified':metadata,'physical_validation_performed':False,'acceptance_errors':clip_errors});errors.extend(member+': '+x for x in clip_errors)
 total=sum(x['frames']for x in rows)
 if len(rows)!=17 or total!=6415:errors.append('Expected17clips/6415frames differs')
 result={'schema':'vita17.independent_input_audit.1','status':'passed_with_diagnostics'if not errors else 'failed','inputs':{'source_archive':{'path':str(args.source_tar),'sha256':archive_sha},'conversion_manifest':{'path':str(manifestpath.relative_to(ROOT)),'sha256':sha(manifestpath)},'urdf':{'path':str(args.urdf.relative_to(ROOT)),'sha256':sha(args.urdf)}},'safety':'stdlib RestrictedReader maps only NumPy dtype/ndarray/reconstruct/scalar to inert local data holders; rejects persistent/extension/custom class globals; no NumPy, Torch, simulator, GPU, or unknown pickle callables used. NPZ parsed as primitive NPY without pickle.','clip_count':len(rows),'total_frames':total,'fps':30,'sum_encoded_video_duration_seconds':sum(x['frames']/x['fps']for x in rows),'all_joint_names':names,'all_conversion_elements_exact':all(x['all_numeric_elements_byte_exact']for x in rows),'raw_pickle_duplicate_groups':[v for v in raw_groups.values()if len(v)>1],'core_motion_numeric_duplicate_groups':[v for v in motion_groups.values()if len(v)>1],'position_violating_cell_count':sum(j['position_below_count']+j['position_above_count']for r in rows for j in r['joint_limit_and_velocity_diagnostics']),'velocity_limit_exceeded_interval_joint_count':sum(j['velocity_limit_exceeded_intervals']for r in rows for j in r['joint_limit_and_velocity_diagnostics']),'position_velocity_checks_are_diagnostics_only':True,'trajectory_modified':False,'physics_steps':0,'physical_validation_passed':False,'recorded_self_collision_refinement_is_not_independent_physical_validation':True,'clips':rows,'acceptance_errors':errors,'elapsed_seconds':time.monotonic()-began};write_json(args.input_report,result);print(json.dumps({k:result[k]for k in ('status','clip_count','total_frames','all_conversion_elements_exact','position_violating_cell_count','velocity_limit_exceeded_interval_joint_count','acceptance_errors','elapsed_seconds')},ensure_ascii=False),flush=True);return not errors
def frac(text):
 a,b=str(text).split('/');return float(a)/float(b)
def audit_videos(args):
 began=time.monotonic();inputaudit=json.loads(args.input_report.read_text(encoding='utf8'));assert inputaudit['status']=='passed_with_diagnostics';manifest=json.loads((args.data_root/'input_manifest.json').read_text(encoding='utf8'));errors=[];rows=[];ffprobe=args.ffprobe or shutil.which('ffprobe');assert ffprobe,'ffprobe is required for CPU full-decode auditing'
 for i,(entry,source)in enumerate(zip(manifest['motions'],inputaudit['clips']),1):
  npzpath=args.data_root/entry['path'];clip=npz(npzpath);names=clip['dof_names'].values();n=clip['dof_pos'].shape[0];fps=clip['fps'].values()[0];source_sha=clip['source_pickle_sha256'].values()[0];stem=f'vita17_{i:02d}_{Path(entry["path"]).stem}';video=args.video_root/(stem+'.mp4');renderpath=args.render_report_root/(stem+'.json');bad=[]
  if not video.is_file()or not renderpath.is_file():
   rows.append({'clip':stem,'status':'missing','video_exists':video.is_file(),'render_report_exists':renderpath.is_file()});errors.append(stem+': missing completed video/report');continue
  render=json.loads(renderpath.read_text(encoding='utf8'));video_sha=sha(video);binding=sha(npzpath)==entry['sha256']==source['npz']['sha256']and source_sha==source['source_pickle_sha256']==entry['source_pickle_sha256'];convention=names==inputaudit['all_joint_names']==render['joint_names_in_source_order']
  if not binding:bad.append('NPZ/source provenance mismatch')
  if not convention:bad.append('52 ordered names mismatch')
  numeric_exact=all(clip[k].c_bytes()and digest(clip[k].c_bytes())==source['numeric_array_checks'][k]['converted_logical_c_sha256']for k in ('root_pos','root_rot_xyzw','dof_pos'))
  if not numeric_exact:bad.append('Numeric/quaternion bytes changed after input audit')
  if sha(args.urdf)!=inputaudit['inputs']['urdf']['sha256']or render.get('urdf_sha256')!=sha(args.urdf):bad.append('URDF differs from independently audited source')
  if render.get('all_source_joint_values_preserved')is not True or render.get('root_xyzw_reordered_to_wxyz')is not True:bad.append('Missing source joint/root convention declaration')
  addresses=render.get('joint_qpos_addresses_in_source_order',[])
  if len(addresses)!=52 or sorted(addresses)!=list(range(7,59)):bad.append('52joint qpos addresses do not cover all joints after free root')
  if render.get('source_sha256')!=sha(npzpath)or render.get('video_sha256')!=video_sha:bad.append('Render-report source/video SHA mismatch')
  if render.get('mode')!='kinematic_replay'or render.get('policy_inference')is not False or render.get('physics_steps')!=0 or render.get('interpolation')is not False or render.get('joint_clipping')is not False:bad.append('Missing/false kinematic-only labels')
  if source_sha not in json.dumps(render.get('source_pickle_bindings',{})):bad.append('Render report missing exact source pickle binding')
  command=[ffprobe,'-v','error','-threads','2','-select_streams','v:0','-count_frames','-show_streams','-show_frames','-show_format','-show_entries','stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration,time_base:frame=best_effort_timestamp_time,pkt_duration_time:format_tags=comment,title,description','-of','json',str(video)];p=subprocess.run(command,capture_output=True,text=True,timeout=180)
  if p.returncode or p.stderr.strip():bad.append('Full-decode ffprobe error: '+p.stderr[-1000:])
  probe=json.loads(p.stdout);stream=probe['streams'][0];frames=probe.get('frames',[]);pts=[float(x['best_effort_timestamp_time'])for x in frames];count=int(stream.get('nb_read_frames',-1));rate=frac(stream['avg_frame_rate']);duration=float(stream.get('duration',len(frames)/fps));spacing=max((abs((b-a)-1/fps)for a,b in zip(pts,pts[1:])),default=0);tags=probe.get('format',{}).get('tags',{})
  if tags.get('comment')!='Kinematic Vita52 replay; no policy or physics simulation; source '+sha(npzpath):bad.append('MP4 kinematic-only labeling/source comment mismatch')
  if count!=n or len(frames)!=n or render.get('source_frames')!=n or render.get('rendered_frames')!=n:bad.append('Complete decoded/source/rendered frame counts differ')
  if abs(rate-fps)>1e-7 or abs(frac(stream['r_frame_rate'])-fps)>1e-7:bad.append('Encoded FPS differs from original30')
  if abs(duration-n/fps)>.005:bad.append('Video duration differs from N/fps')
  if not pts or abs(pts[0])>1e-5 or any(b<=a for a,b in zip(pts,pts[1:]))or spacing>5e-6:bad.append('Presentation timestamps not complete original constant-frame-rate sequence')
  rows.append({'clip':stem,'status':'passed'if not bad else 'failed','video':{'path':str(video.relative_to(ROOT)),'sha256':video_sha,'bytes':video.stat().st_size},'render_report':{'path':str(renderpath.relative_to(ROOT)),'sha256':sha(renderpath)},'source_pickle_sha256':source_sha,'npz_sha256':sha(npzpath),'all52_names_exact':convention,'original_numeric_and_xyzw_bytes_still_exact':bool(numeric_exact),'source_provenance_bindings_exact':binding,'source_frames':n,'decoded_frames':count,'enumerated_presentation_frames':len(frames),'fps':rate,'duration_seconds':duration,'expected_duration_seconds':n/fps,'first_pts_seconds':pts[0]if pts else None,'last_pts_seconds':pts[-1]if pts else None,'maximum_pts_spacing_error_seconds':spacing,'decoded_stream':stream,'mp4_label_tags':tags,'ffprobe_software_full_decode_returncode':p.returncode,'kinematic_labels_verified':not any('label'in x for x in bad),'acceptance_errors':bad});errors.extend(stem+': '+x for x in bad);print(json.dumps({'clip':i,'frames':count,'status':rows[-1]['status'],'errors':bad}),flush=True)
 expected_total=sum(x['frames']for x in inputaudit['clips']);total=sum(x.get('decoded_frames',0)for x in rows)
 if sha(args.data_root/'input_manifest.json')!=inputaudit['inputs']['conversion_manifest']['sha256']:errors.append('Conversion manifest changed after independent input audit')
 if len(rows)!=17 or total!=expected_total or total!=6415:errors.append('Final complete17/6415 coverage not met')
 result={'schema':'vita17.independent_video_audit.1','status':'passed'if not errors else 'failed_or_incomplete','input_audit':{'path':str(args.input_report.relative_to(ROOT)),'sha256':sha(args.input_report)},'conversion_manifest':{'path':str((args.data_root/'input_manifest.json').relative_to(ROOT)),'sha256':sha(args.data_root/'input_manifest.json')},'expected_clips':17,'completed_clips':sum(x['status']=='passed'for x in rows),'decoded_total_frames':total,'expected_total_frames':6415,'fps':30,'audit_device':'CPU','ffprobe_full_frame_decode':True,'policy_inference':False,'physics_steps':0,'physical_validation_passed':False,'rendered_pixel_poses_are_not_independently_3d_reconstructed':'Input exactness is independently audited; renderer52joint mapping and metadata are checked; video checks prove full decoding/time/count, not pixel-based recovery of joint angles.','clips':rows,'acceptance_errors':errors,'elapsed_seconds':time.monotonic()-began};write_json(args.video_report,result);print(json.dumps({k:result[k]for k in ('status','completed_clips','decoded_total_frames','acceptance_errors','elapsed_seconds')},ensure_ascii=False),flush=True);return not errors
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('phase',choices=['inputs','videos']);ap.add_argument('--data-root',type=Path,default=ROOT/'data/vita_demo_20261026');ap.add_argument('--source-tar',type=Path,default=SOURCE_TAR);ap.add_argument('--urdf',type=Path);ap.add_argument('--input-report',type=Path,default=ROOT/'reports/vita17-input-audit.json');ap.add_argument('--video-root',type=Path,default=ROOT/'outputs/previews');ap.add_argument('--render-report-root',type=Path,default=ROOT/'reports/vita_motion_demos');ap.add_argument('--video-report',type=Path,default=ROOT/'reports/vita17-video-audit.json');ap.add_argument('--ffprobe');args=ap.parse_args();args.urdf=args.urdf or args.data_root/'robot/vita02a/urdf/vita02a_linkerhand_meshcoll_relative.urdf';ok=audit_inputs(args)if args.phase=='inputs'else audit_videos(args);raise SystemExit(0 if ok else 2)
if __name__=='__main__':main()
