"""Render actual TraCI snapshots around the gate-demand reversal."""
import argparse
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
from nchu_env import GateEnv
from nchu_demand import ARMS
from nchu_test_policies import profile_file, load_models

def render(models, out):
    out.mkdir(parents=True, exist_ok=True)
    model = next(m for name, seed, m in load_models(models) if (name,seed)==("DQN",11))
    env = GateEnv(profile_file("reversal",out/"inputs"))
    fonts = [ImageFont.truetype("DejaVuSans.ttf",size) for size in (16,20,28)]
    try:
        obs,_ = env.reset(seed=701)
        net = ET.parse(env.directory/"gate.net.xml")
        center = net.find("junction[@id='gate']")
        cx,cy = float(center.get("x")),float(center.get("y"))
        def project(x,y):
            return 310+(x-cx)*3.1,270-(y-cy)*3.1
        base = Image.new("RGB",(960,540),"#eef2ee")
        draw = ImageDraw.Draw(base)
        for lane in net.findall(".//edge/lane"):
            points=[project(*map(float,p.split(",")[:2])) for p in lane.get("shape").split()]
            draw.line(points, fill="#b3afa4" if lane.get("allow")=="pedestrian" else "#747e84",width=max(2,round(float(lane.get("width","3.2"))*3.1)))
        cmd=["ffmpeg","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","960x540","-r","2","-i","-","-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",str(out/"Demo.mp4")]
        with subprocess.Popen(cmd,stdin=subprocess.PIPE) as pipe:
            try:
                while env.time < 1050:
                    action=int(model.predict(obs,deterministic=True)[0])
                    obs,*_=env.step(action)
                    if env.time<=750:
                        continue
                    frame=base.copy()
                    d=ImageDraw.Draw(frame)
                    for domain in (env.conn.vehicle,env.conn.person):
                        for obj in domain.getIDList():
                            x,y=project(*domain.getPosition(obj))
                            color="#e7af34" if domain is env.conn.person else ("#50d5c4" if "motor" in domain.getTypeID(obj) else "#2880db")
                            d.ellipse((x-3,y-3,x+3,y+3),fill=color,outline="#192f40")
                    d.rectangle((600,0,960,540),fill="#132e42")
                    def label(y,s,size=1,color="white"):
                        d.text((625,y),s,font=fonts[size],fill=color)
                    label(26,"NCHU main gate",2)
                    label(77,"Gate reversal  /  DQN seed 11")
                    label(110,"Synthetic demand  /  traffic 701",0)
                    label(151,f"Simulation clock   {env.time//60:02}:{env.time%60:02}")
                    phase=env.signal.phase
                    label(191,("EW" if phase<4 else "NS")+"   "+["green","ped. clearance","yellow","all red"][phase%4])
                    label(233,"Queued vehicles by approach",0)
                    label(260,"   ".join(f"{a} {q}" for a,q in zip(ARMS,env.queues)))
                    label(326,"Vehicle",0,"#64aafa")
                    label(355,"Motorcycle",0,"#50d5c4")
                    label(384,"Pedestrian",0,"#e7af34")
                    label(442,"North up  /  playback 10x",0)
                    label(477,"Roads: OpenStreetMap contributors",0)
                    if env.time==900:
                        frame.save(out/"intersection.png")
                    pipe.stdin.write(frame.tobytes())
            finally:
                pipe.stdin.close()
            if pipe.wait():
                raise RuntimeError("Video encoding failed")
    finally:
        env.close()

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("models",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    render(args.models,args.out)
