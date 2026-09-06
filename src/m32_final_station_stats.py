"""Final M68 MS-GRAN station uncertainty, tail-error, and mechanism statistics."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from m6_multiscale_correlation_diagnostic import causal_ewma, component_correlations

ROOT=Path(__file__).resolve().parents[1]; REPORTS=ROOT/'reports'; DATA=ROOT/'data_processed'/'ngl_cascadia_90stations_2010-2024_residuals_v1.npz'

def station_rmse(pred,truth,mask):
    valid=mask[...,None] & np.isfinite(pred) & np.isfinite(truth)
    sq=np.where(valid,(pred-truth)**2,0.0)
    return np.sqrt(sq.sum((0,1,3))/np.maximum(valid.sum((0,1,3)),1))
def paired_bootstrap(rb,rg,rng,n=10000):
    vals=[]
    for _ in range(n):
        ix=rng.integers(0,len(rb),len(rb)); vals.append(100*(1-rg[ix].mean()/rb[ix].mean()))
    return [float(x) for x in np.percentile(vals,[2.5,97.5])]
def tail(pred,truth,mask):
    v=mask[...,None] & np.isfinite(pred) & np.isfinite(truth); e=np.abs(pred[v]-truth[v])
    return {'samples':int(e.size),'mae_mm':float(e.mean()),'p90_mm':float(np.percentile(e,90)),'p95_mm':float(np.percentile(e,95)),'p99_mm':float(np.percentile(e,99)),'maximum_mm':float(e.max())},e
def ranks(x):
    order=np.argsort(x); r=np.empty(len(x),float); r[order]=np.arange(len(x)); return r
def spearman(x,y): return float(np.corrcoef(ranks(np.asarray(x)),ranks(np.asarray(y)))[0,1])
def haversine(lat1,lon1,lat2,lon2):
    p1,p2=np.radians(lat1),np.radians(lat2); dp=np.radians(lat2-lat1); dl=np.radians(lon2-lon1)
    a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2; return 6371*2*np.arcsin(np.sqrt(a))
def grouped(name,feature,gain):
    cuts=np.quantile(feature,[1/3,2/3]); labels=np.digitize(feature,cuts); out=[]; rng=np.random.default_rng(20260804)
    for k,label in enumerate(('low','medium','high')):
        ix=np.where(labels==k)[0]; boots=[float(np.mean(gain[rng.choice(ix,len(ix),replace=True)])) for _ in range(10000)]
        out.append({'group':label,'stations':int(len(ix)),'feature_min':float(feature[ix].min()),'feature_max':float(feature[ix].max()),'mean_gain_percent':float(gain[ix].mean()),'median_gain_percent':float(np.median(gain[ix])),'improved_stations':int((gain[ix]>0).sum()),'bootstrap_ci_percent':[float(v) for v in np.percentile(boots,[2.5,97.5])]})
    return {'factor':name,'tertile_boundaries':[float(v) for v in cuts],'spearman_with_station_gain':spearman(feature,gain),'groups':out}
def main():
    per_seed=[]; station_gains=[]; tails=[]
    for k in range(1,4):
        b=np.load(ROOT/'experiments'/f'M28_nlinear_1day_90stations_seed{k}'/'real_test_forecasts.npz'); g=np.load(ROOT/'experiments'/f'M68_msgran_1day_90stations_seed{k}'/'real_test_forecasts.npz')
        truth=b['truth_enu_mm']; mask=b['observed_mask']; rb=station_rmse(b['prediction_enu_mm'],truth,mask); rg=station_rmse(g['prediction_enu_mm'],truth,mask); gain=100*(1-rg/rb); station_gains.append(gain)
        tb,eb=tail(b['prediction_enu_mm'],truth,mask); tg,eg=tail(g['prediction_enu_mm'],truth,mask); threshold=tb['p95_mm']; tg['fraction_above_nlinear_test_p95']=float(np.mean(eg>threshold)); tb['fraction_above_own_p95']=float(np.mean(eb>threshold))
        tails.append({'seed':k,'nlinear':tb,'msgran':tg,'relative_reduction_percent':{q:100*(1-tg[q]/tb[q]) for q in ('mae_mm','p90_mm','p95_mm','p99_mm','maximum_mm')}})
        per_seed.append({'seed':k,'improved_stations':int((gain>0).sum()),'mean_gain_percent':float(gain.mean()),'median_gain_percent':float(np.median(gain)),'max_gain_percent':float(gain.max()),'min_gain_percent':float(gain.min()),'bootstrap_ci_percent':paired_bootstrap(rb,rg,np.random.default_rng(20260800+k))})
    gain=np.mean(station_gains,axis=0); src=np.load(DATA); raw=src['residual_enu_mm'].astype(np.float32); obs=src['observed_mask'].astype(bool); split=src['split']; _,high=causal_ewma(raw,obs,31); train=split==0; corr=component_correlations(high[train],obs[train]); np.fill_diagonal(corr,np.nan)
    corr_strength=np.nanmean(np.sort(np.abs(corr),axis=1)[:,-8:],axis=1); lat,lon=src['latitude'],src['longitude']; distances=[]
    for i in range(len(lat)):
        ids=np.argsort(np.nan_to_num(np.abs(corr[i]),nan=-1))[-8:]; distances.append(float(np.mean([haversine(lat[i],lon[i],lat[j],lon[j]) for j in ids])))
    sigma=src['sigma_enu_mm'][train]; noise=np.nanmedian(np.linalg.norm(sigma,axis=2),axis=0)
    payload={'definition_notes':{'tail':'Frozen-test descriptive absolute-error quantiles; not anomaly-label false-alarm rates.','grouping':'Tertiles defined from training-period features only.','bootstrap':'Paired station resampling, 10000 replicates; stations are the resampling unit.'},'station_bootstrap':per_seed,'tail_error':tails,'mechanism_groups':[grouped('training_top8_absolute_correlation',corr_strength,gain),grouped('mean_distance_to_top8_correlated_neighbors_km',np.asarray(distances),gain),grouped('training_median_formal_noise_norm_mm',noise,gain)]}
    (REPORTS/'M82_MSGRAN_station_tail_mechanism_statistics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(REPORTS/'M82_MSGRAN_station_tail_mechanism_statistics.json')
if __name__=='__main__': main()
