"""Extract an unmodified RGB contact sheet from the new source video."""
from pathlib import Path
import argparse,io,subprocess
def main():
    p=argparse.ArgumentParser();p.add_argument('--video',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import imageio_ffmpeg
    from PIL import Image,ImageDraw
    canvas=Image.new('RGB',(1280,1520),'#161b22');draw=ImageDraw.Draw(canvas)
    for i,frame in enumerate([0,90,120,150,180,210,270,349]):
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-threads','1','-ss',str(frame/30),'-i',str(a.video),'-frames:v','1','-vf','scale=640:360','-f','image2pipe','-vcodec','mjpeg','pipe:1']
        run=subprocess.run(cmd,capture_output=True,check=True);im=Image.open(io.BytesIO(run.stdout)).convert('RGB');x=(i%2)*640;y=(i//2)*380;canvas.paste(im,(x,y+20));draw.text((x+8,y+3),f'RAW RGB | frame {frame} | t={frame/30:.3f}s',fill='white')
    a.output.parent.mkdir(parents=True,exist_ok=True);canvas.save(a.output,quality=94);print(a.output)
if __name__=='__main__':main()
