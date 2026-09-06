"""Three-seed 3/7-day matched NLinear and graph-adapter experiments."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; T=ROOT/'src'/'m22_train_strong_baseline.py'; E=ROOT/'src'/'m22_evaluate_strong_baseline.py'; AT=ROOT/'src'/'m26_train_dlinear_graph_adapter.py'; AE=ROOT/'src'/'m26_evaluate_dlinear_graph_adapter.py'; SEEDS=(20260803,20260804,20260805)
def run(c): print('[M31]',' '.join(c),flush=True); subprocess.run(c,cwd=ROOT,check=True)
def metric(p,e,device):
    f=p/'real_test_forecast_metrics.json'
    if not f.exists(): run([sys.executable,str(e),'--checkpoint',str(p/'best.pt'),'--device',device,'--save-forecasts'])
    return json.loads(f.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int,default=20);p.add_argument('--resume',action='store_true');a=p.parse_args(); rows=[]
 for h in (3,7):
  for idx,seed in enumerate(SEEDS,1):
   b=ROOT/'experiments'/f'M31_nlinear_{h}day_seed{idx}'; fm=b/'real_test_forecast_metrics.json'
   if not(a.resume and fm.exists()): run([sys.executable,str(T),'--model','nlinear','--horizon',str(h),'--epochs',str(a.epochs),'--seed',str(seed),'--device',a.device,'--run-name',b.name])
   bm=metric(b,E,a.device)
   g=ROOT/'experiments'/f'M31_nlinear_graph_{h}day_seed{idx}'; gm=g/'real_test_forecast_metrics.json'
   if not(a.resume and gm.exists()): run([sys.executable,str(AT),'--base-checkpoint',str(b/'best.pt'),'--run-name',g.name,'--epochs',str(a.epochs),'--seed',str(seed),'--device',a.device])
   am=metric(g,AE,a.device)
   rows.append({'horizon_days':h,'seed':seed,'nlinear_rmse_mm':bm['model']['overall']['rmse_mm'],'adapter_rmse_mm':am['model']['overall']['rmse_mm'],'adapter_skill_percent':100*am['overall_rmse_skill_vs_persistence'],'gain_vs_nlinear_percent':100*(1-am['model']['overall']['rmse_mm']/bm['model']['overall']['rmse_mm'])})
 out=[]
 for h in (3,7):
  x=[r for r in rows if r['horizon_days']==h]; q={'horizon_days':h,'seeds':len(x)}
  for k in ('nlinear_rmse_mm','adapter_rmse_mm','adapter_skill_percent','gain_vs_nlinear_percent'):
   z=np.array([r[k] for r in x]);q[k+'_mean']=float(z.mean());q[k+'_std']=float(z.std(ddof=1))
  out.append(q)
 (ROOT/'reports'/'M31_final_multihorizon_3seed.json').write_text(json.dumps({'runs':rows,'summary':out},indent=2),encoding='utf-8')
if __name__=='__main__':main()
