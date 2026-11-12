"""Read-only checks of prototype delivery links, files, ZIP and existing videos."""
from pathlib import Path
import argparse,hashlib,json,re,subprocess,zipfile
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--archive',type=Path)
    p.add_argument('--video-check',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args();r=a.root.resolve()
    m=json.loads((r/'delivery_manifest.json').read_text(encoding='utf-8'));bad=[]
    for entry in m['files']:
        f=(r/entry['path']).resolve()
        if not f.is_relative_to(r)or not f.is_file()or sha(f)!=entry['sha256']:bad.append(entry['path'])
    links=[]
    for file in ['index.html','report/report.html','report/REPORT.zh-CN.md']:
        source=r/file;text=source.read_text(encoding='utf-8');targets=re.findall(r'\]\(([^)]+)\)',text)+re.findall(r'(?:src|href)="([^\"]+)"',text)
        for target in targets:
            if target.startswith(('http:','https:','#')):continue
            f=(source.parent/target).resolve();links.append({'source':file,'target':target,'exists':f.is_relative_to(r)and f.exists()})
    final=m['final_selection_record'];checkpoint=r/Path(final['selected_checkpoint']).relative_to(Path(m['source_experiment']))
    selected_ok=checkpoint.is_file()and sha(checkpoint)==final['selected_checkpoint_sha256']
    zip_result=None
    if a.archive:
        with zipfile.ZipFile(a.archive)as z:
            crc=z.testzip();embedded=json.loads(z.read('delivery_manifest.json').decode('utf-8'));archive_bad=[]
            for entry in embedded['files']:
                h=hashlib.sha256()
                with z.open(entry['path'])as f:
                    for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
                if h.hexdigest()!=entry['sha256']:archive_bad.append(entry['path'])
            same_manifest=hashlib.sha256(z.read('delivery_manifest.json')).hexdigest()==sha(r/'delivery_manifest.json')
        zip_result={'sha256':sha(a.archive),'bytes':a.archive.stat().st_size,'crc_passed':crc is None,
            'manifest_matches_extracted':same_manifest,'file_hashes_checked':len(embedded['files']),'bad_file_hashes':archive_bad}
    videos=[]
    if a.video_check:
        for name in ['sonic_all_methods','terminal_refined_full_v4','expert_control_v1','ppo_feedback_prototype']:
            f=r/'media'/(name+'.mp4');probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration','-of','json',str(f)],capture_output=True,text=True)
            decode=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(f),'-f','null','-'],capture_output=True,text=True)
            videos.append({'name':name,'sha256':sha(f),'ffprobe':json.loads(probe.stdout),'probe_returncode':probe.returncode,
                'all_frames_decode_returncode':decode.returncode,'decode_stderr':decode.stderr})
    broken=[x for x in links if not x['exists']]
    result={'status':'passed','delivery_status':m['status'],'file_hashes_checked':len(m['files']),'bad_file_hashes':bad,
        'links_checked':len(links),'broken_links':broken,'selected_checkpoint_sha_matches':selected_ok,'archive':zip_result,
        'videos':videos,'no_physics_or_training_executed':True,'does_not_claim_new_rl_gain':not final['rl_incremental_gain_proven']}
    if bad or broken or not selected_ok or(zip_result and(not zip_result['crc_passed']or not zip_result['manifest_matches_extracted']or zip_result['bad_file_hashes']))or any(x['probe_returncode']or x['all_frames_decode_returncode']for x in videos):result['status']='failed'
    if a.output:a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['status']!='passed':raise RuntimeError('Delivery validation failed')
if __name__=='__main__':main()
