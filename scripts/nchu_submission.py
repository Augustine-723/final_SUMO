"""Build the English final report, slide deck and self-contained source archive."""
import argparse
import csv
import hashlib
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
profiles = ["offpeak","balanced","directional","reversal","ped_peak"]
names = ["Off-peak","Balanced","East-heavy","Gate reversal","Pedestrian peak"]
authors = "林瑞裕、官奕愷"

def rows(file):
    with file.open() as f:
        return list(csv.DictReader(f))

class Document:
    def __init__(self, file, slides):
        self.w,self.h=(960,540) if slides else A4
        self.slide=slides
        self.c=Canvas(str(file),pagesize=(self.w,self.h))
        self.c.setTitle("Traffic Signal Optimization at the NCHU Main Gate")
        self.c.setAuthor(authors)
        self.x=46
        self.width=self.w-2*self.x
        self.page=0

    def heading(self,title):
        self.page+=1
        self.y=self.h-40
        self.para(title,30 if self.slide else 22,True)
        self.c.setStrokeColor(colors.HexColor("#19867b"))
        self.c.line(self.x,self.y,self.w-self.x,self.y)
        self.y-=20

    def para(self,s,size=None,bold=False):
        size=size or (19 if self.slide else 11.5)
        p=Paragraph(escape(s),ParagraphStyle("body",fontName="Helvetica-Bold" if bold else "Helvetica",
                    fontSize=size,leading=size*1.33,textColor=colors.HexColor("#162f41")))
        _,height=p.wrap(self.width,1000)
        self.y-=height
        p.drawOn(self.c,self.x,self.y)
        self.y-=12 if self.slide else 9

    def team(self):
        im=Image.new("RGB",(410,60),"white")
        font=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",40,index=3)
        ImageDraw.Draw(im).text((1,0),authors,fill="#162f41",font=font)
        self.y-=32
        self.c.drawImage(ImageReader(im),self.x,self.y,219,32)
        self.y-=15

    def figure(self,file,maxheight):
        img=ImageReader(str(file))
        w,h=img.getSize()
        scale=min(self.width/w,maxheight/h)
        self.y-=scale*h
        self.c.drawImage(img,(self.w-w*scale)/2,self.y,w*scale,h*scale)
        self.y-=10

    def table(self,headers,values):
        weights=[1.65]+[1.]*(len(headers)-1)
        if headers[-1]=="Demand pattern":
            weights=[1.4,.8,.8,2.25]
        cols=[self.x]
        for weight in weights:
            cols.append(cols[-1]+self.width*weight/sum(weights))
        height=32 if self.slide else 24
        for i,row in enumerate([headers,*values]):
            self.y-=height
            self.c.setFillColor(colors.HexColor("#193f51" if i==0 else ("#e8f1ef" if i%2 else "#f7f9f9")))
            self.c.rect(self.x,self.y,self.width,height,stroke=0,fill=1)
            self.c.setFillColor(colors.white if i==0 else colors.HexColor("#193648"))
            self.c.setFont("Helvetica-Bold" if i==0 else "Helvetica",15 if self.slide else 9.8)
            for j,cell in enumerate(row):
                self.c.drawString(cols[j]+6,self.y+height*.32,str(cell))
        self.y-=13

    def end(self):
        if self.y<28:
            raise ValueError(f"Content overflow on page {self.page}: {self.y}")
        self.c.setFillColor(colors.HexColor("#466271"))
        self.c.setFont("Helvetica",9)
        self.c.drawString(self.x,20,"Reinforcement Learning   /   NCHU main gate")
        self.c.drawRightString(self.w-self.x,20,str(self.page))
        self.c.showPage()

def build(analysis, models, video, out):
    out.mkdir(parents=True,exist_ok=True)
    summary=rows(analysis/"summary.csv")
    pairs=rows(analysis/"paired.csv")
    metrics={(r["scenario"],r["method"],r["metric"]):float(r["mean"]) for r in summary}
    wait="mean_vehicle_wait_s"
    ped="mean_pedestrian_wait_s"
    def performance(metric):
        return [[n,*[f"{metrics[p,m,metric]:.2f}" for m in ("fixed","actuated","DQN","PPO")]] for p,n in zip(profiles,names)]
    def comparison(m,ref,metric):
        rr=[r for r in pairs if (r["method"],r["reference"],r["metric"])==(m,ref,metric)]
        return [[n,f"{float(r['delta']):+.2f}",f"{float(r['ci_low']):+.2f} to {float(r['ci_high']):+.2f}",f"{float(r['change_pct']):+.1f}%"] for n,r in zip(names,rr)]
    traffic=[["Off-peak","148","36","Training profile, fresh seeds"],["Balanced","400","112","Equal counts on all arms"],["East-heavy","306","112","East arm carries 90 then 96"],["Gate reversal","262","112","North peak shifts to south"],["Pedestrian peak","400","336","Triple the balanced walk count"]]
    headers=["Profile","Fixed","Actuated","DQN","PPO"]
    intro="At the NCHU main gate, DQN reduced mean vehicle waiting relative to fixed timing in four synthetic profiles. A queue-actuated controller had the lowest mean vehicle waiting among the four core methods on every profile."
    site="The model follows OpenStreetMap road geometry at Xingda Road and Xuefu Road, Taichung (24.12412 N, 120.67515 E). The south arm represents the campus gate. Lane count, speed, turn shares and signal timing are assumptions. Available municipal detector locations did not supply four-arm turning counts at this gate."
    design="Each profile uses two 15-minute demand bins followed by 15 minutes for clearance. Seeds 701-720 randomize departure times within bins. Every controller receives the same route file for each profile and seed. The off-peak volumes occurred in training; the other four demand profiles were reserved for testing."
    learn="DQN learns an action-value function. PPO uses clipped policy updates and a value estimate. Each core method trained three 64-by-64 MLPs for 30 episodes (16,200 decisions) on shuffled off-peak, morning and evening profiles. Training seeds were 11, 22 and 33. Deterministic evaluation used the final checkpoint of each run."
    mdp="The controller observes four inbound queues, four vehicle counts, an eight-stage signal indicator, elapsed phase time, episode time and crossing occupancy. Every five seconds it requests an east-west or north-south green group. Core reward is minus the sum of queued vehicle-seconds during that decision."
    safety="The signal enforces 10-45 seconds of normal green. A switch passes through five seconds of pedestrian clearance, three seconds of yellow and two seconds of all-red, extended if a crossing is occupied. Fixed timing requests a switch after 20 seconds of normal green. Actuated control requests the opposing group when its queue is longer."
    stats="We average the three frozen policies within each traffic seed for each core RL method. Intervals resample 20 paired traffic-seed differences 10,000 times. They describe demand-seed variability conditional on the trained models. The exploratory profile comparisons use unadjusted 95% percentile intervals."
    bias="SUMO recorded 152 unfinished vehicle trips, all from east-heavy PPO seed 22, and 18 vehicle teleports across the full matrix. No collisions, unfinished pedestrian trips or pedestrian teleports were recorded. Vehicle wait and travel time cover completed trips; severe queues can therefore suppress the reported average."
    behavior="Under east-heavy demand, individual PPO training seeds 11, 22 and 33 produced mean waits of 12.31, 150.22 and 11.77 seconds. Seed 22 also left 7.6 vehicles unfinished per episode on average. The three-policy mean of 58.10 seconds masks that instability."
    ablate="With PPO training seed 11 held fixed, adding a 0.25 penalty per stopped pedestrian-second changed pedestrian stopped time by -0.19 s in the reversal and +0.42 s in the pedestrian peak. The latter interval spans zero. Vehicle wait in the pedestrian peak changed by +0.63 s (95% interval -0.89 to +3.27)."
    lowlr="Reducing PPO's learning rate from 0.0003 to 0.0001 at seed 11 raised vehicle wait in all five profiles, by 7.65 to 9.57 seconds. Both ablations cover a single training seed."
    nextstep="Collect synchronized turning counts, crosswalk volumes, signal timings and per-arm queues at this gate. Fit the network on one survey period and validate on a separate period. Repeat the reward ablation across training seeds and report incomplete trips alongside travel times."
    for slide in [False,True]:
        d=Document(out/("Presentation.pdf" if slide else "Report.pdf"),slide)
        d.heading("Traffic Signal Optimization at the NCHU Main Gate")
        d.para("Frozen DQN and PPO policies under changing vehicle and pedestrian demand",bold=True)
        d.team()
        if not slide:
            d.para("Abstract",bold=True)
            d.para(intro+" The experiment covers 1,000 simulated episodes. The pedestrian-reward pilot showed no clear gain during the pedestrian peak. Site geometry is grounded in map data; all traffic volumes are synthetic.")
        d.figure(video/"intersection.png",225 if slide else 245)
        d.para("SUMO replay of the north-to-south demand reversal. Road geometry © OpenStreetMap contributors.",12 if slide else 9)
        d.end()
        if slide:
            d.heading("Research questions")
            for s in ["RQ1  Can DQN or PPO shorten vehicle waiting relative to fixed timing?","RQ2  Does performance persist when demand changes direction or intensity?","RQ3  Does explicitly penalizing pedestrian stops improve walking outcomes at a cost to vehicles?"]:
                d.para(s)
            d.para("Study scope: one mapped intersection and controlled, synthetic demand.",bold=True)
            d.end()
        d.heading("Intersection and traffic demand")
        d.para(site)
        d.table(["Profile","Vehicles","Walkers","Demand pattern"],traffic) if slide else None
        if not slide:
            d.para("RQ1 compares waiting against fixed timing. RQ2 tests generalization under directional and volume shifts. RQ3 compares vehicle-only and pedestrian-aware PPO rewards.")
            d.table(["Profile","Vehicles","Walkers","Demand pattern"],traffic)
        d.para("Counts are scheduled totals over 30 minutes. The pedestrian peak has the same vehicle counts as the balanced profile.",14 if slide else 10)
        if not slide:
            d.para(design)
            d.para("One mixed lane per direction serves cars and motorcycles at an assumed maximum of 30 km/h. Four modeled crossings serve one walking direction per arm. Gate admission, surrounding intersections and upstream spillback are outside the model.")
        d.end()
        d.heading("Signal control and learning")
        d.para(mdp)
        d.para(safety)
        if not slide:
            d.para(learn)
            d.para("The two training variants change only the PPO learning rate or pedestrian reward coefficient. Validation used traffic seeds 101 and 102. Training schedules, model checksums, versions and per-episode histories accompany the source archive.")
        d.end()
        d.heading("Training curves")
        d.figure(models/"learning_curves.png",305 if slide else 355)
        d.para("Returns include different demand profiles. A reward with an added pedestrian penalty has a different numerical scale.",15 if slide else 10)
        if not slide:
            d.para("Evaluation protocol",bold=True)
            d.para(stats)
            d.para("Primary metric: SUMO stopped waiting seconds per completed motor vehicle. Secondary endpoints: total inbound queue averaged across 2,700 seconds, completed-trip duration, arrivals and seconds below 0.1 m/s per completed walker. A pedestrian stop can occur away from the curb.")
        d.end()
        if slide:
            d.heading("Paired evaluation")
            d.para(learn)
            d.para(design)
            d.para(stats,16)
            d.end()
        d.heading("Vehicle waiting time by controller")
        d.table(headers,performance(wait))
        d.para("Mean seconds per completed vehicle. Lower values indicate shorter waiting.",14 if slide else 10)
        d.para(intro)
        if not slide:
            d.para("DQN minus fixed timing",bold=True)
            d.table(["Profile","Delta (s)","95% interval (s)","Change"],comparison("DQN","fixed",wait))
            d.para("The largest relative decrease was off-peak. During the reversal, DQN and fixed timing differed by less than 0.01 s. The vehicle ranking therefore depends on the incoming traffic pattern.")
        d.end()
        if slide:
            d.heading("How stable was the DQN improvement?")
            d.table(["Profile","Delta (s)","95% interval (s)","Change"],comparison("DQN","fixed",wait))
            d.para("DQN minus fixed timing, paired by traffic seed. In the reversal, the interval includes both a benefit and a cost.")
            d.end()
        if slide:
            d.heading("Vehicle and pedestrian waiting across profiles")
            d.figure(analysis/"comparison.png",344)
            d.para("Each bar averages 20 traffic seeds. Core RL bars also average three frozen models.",15)
            d.end()
        d.heading("Generalization and incomplete trips")
        d.para(behavior)
        d.para(bias)
        if not slide:
            for m,t in [("mean_total_queue","Mean total queue (vehicles)"),("mean_vehicle_trip_s","Completed vehicle trip duration (seconds)"),("vehicle_arrivals","Completed vehicles per episode")]:
                d.para(t,bold=True)
                d.table(headers,performance(m))
        d.end()
        d.heading("Pedestrian outcomes and reward ablation")
        d.table(["Profile","Delta (s)","95% interval (s)","Change"],comparison("PPO_pedestrian","PPO_seed11",ped))
        d.para("Pedestrian-aware minus vehicle-only PPO, both trained with seed 11. Negative values indicate less stopped time.",15 if slide else 10)
        d.para(ablate,16 if slide else None)
        if not slide:
            d.table(headers,performance(ped))
            d.para("Core methods: mean stopped seconds per completed walker.",10)
            d.para(lowlr)
        d.end()
        if slide:
            d.heading("Learning-rate sensitivity")
            d.table(["Profile","Delta (s)","95% interval (s)","Change"],comparison("PPO_low_lr","PPO_seed11",wait))
            d.para(lowlr)
            d.end()
        d.heading("Conclusions and limits")
        d.para("RQ1. DQN shortened waiting under four profiles. The simple queue-actuated controller remains the strongest vehicle-wait baseline in this experiment.")
        d.para("RQ2. A reversal removed the DQN advantage over fixed timing. Core PPO improved on fixed timing there, but one training seed performed poorly under east-heavy demand.")
        d.para("RQ3. The reward ablation showed a small pedestrian benefit during reversal. Its effect during the pedestrian peak remains uncertain.")
        d.para(nextstep,16 if slide else None)
        if not slide:
            d.para("Sources and replication",bold=True)
            d.para("OpenStreetMap contributors, node 13021816568 (openstreetmap.org/node/13021816568). Taichung detector inventory (data.gov.tw/dataset/84657). SUMO (sumo.dlr.de) and Stable-Baselines3 (stable-baselines3.readthedocs.io).")
            d.para("Repository: github.com/Augustine-723/final_SUMO. Training run 45134925800; frozen evaluation run 45137746130. Source_Code.zip includes raw episode metrics, matched contrasts, run metadata, checkpoints and executable workflows.")
        d.end()
        d.c.save()
    shutil.copy2(video/"Demo.mp4",out/"Demo.mp4")
    with ZipFile(out/"Source_Code.zip","w",ZIP_DEFLATED) as z:
        manifest=ROOT/"SOURCE_FILES.txt"
        files=manifest.read_text().splitlines() if manifest.exists() else subprocess.check_output(["git","ls-files","-z"],cwd=ROOT).decode().split("\0")
        files=[f for f in files if f and not f.startswith("submission/")]
        z.writestr("SOURCE_FILES.txt","\n".join(files)+"\n")
        for file in files:
            z.write(ROOT/file,file)
        for folder,label in [(analysis,"results/phase7"),(models,"results/phase6")]:
            for file in sorted(folder.rglob("*")):
                if file.is_file():
                    z.write(file,str(Path(label)/file.relative_to(folder)))
        z.write(video/"intersection.png","results/demo/intersection.png")
    files=["Report.pdf","Presentation.pdf","Demo.mp4","Source_Code.zip"]
    (out/"SHA256SUMS").write_text("".join(hashlib.sha256((out/f).read_bytes()).hexdigest()+"  "+f+"\n" for f in files))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("analysis","models","video","out"):
        parser.add_argument(key,type=Path)
    build(**vars(parser.parse_args()))
