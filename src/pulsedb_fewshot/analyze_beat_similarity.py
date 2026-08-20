"""Within-window beat-to-beat morphology similarity for 10-second PPG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .training import WaveformAccessor, load_store_metadata, save_json


def _peaks(signal:np.ndarray,sampling_rate:float)->np.ndarray:
    smooth=np.convolve(signal,np.ones(5)/5,mode="same"); candidate=np.flatnonzero((smooth[1:-1]>smooth[:-2])&(smooth[1:-1]>=smooth[2:]))+1; candidate=candidate[smooth[candidate]>=np.median(smooth)+0.05*np.std(smooth)]
    minimum=max(1,int(round(0.30*sampling_rate))); selected=[]
    for index in candidate[np.argsort(smooth[candidate])[::-1]]:
        if all(abs(int(index)-other)>=minimum for other in selected): selected.append(int(index))
    return np.asarray(sorted(selected),dtype=int)


def _beats(signal:np.ndarray,sampling_rate:float)->tuple[list[np.ndarray],np.ndarray]:
    peak=_peaks(signal,sampling_rate)
    if len(peak)<4:return [],peak
    trough=np.asarray([peak[i]+int(np.argmin(signal[peak[i]:peak[i+1]+1])) for i in range(len(peak)-1)],dtype=int); beats=[]
    for start,stop in zip(trough[:-1],trough[1:]):
        duration=(stop-start)/sampling_rate
        if not 0.30<=duration<=2.0 or stop-start<4:continue
        values=np.interp(np.linspace(0,stop-start,100),np.arange(stop-start+1),signal[start:stop+1]); sd=float(values.std())
        if np.isfinite(sd) and sd>1e-8: beats.append(((values-values.mean())/sd).astype(np.float32))
    return beats,peak


def beat_similarity(signal:np.ndarray,sampling_rate:float=125.0)->dict[str,float|int|str|bool]:
    x=np.asarray(signal,dtype=np.float64).reshape(-1)
    if len(x)<sampling_rate*3 or not np.isfinite(x).all() or float(x.std())<=1e-8:return {"valid":False,"reason":"invalid_signal","n_beats":0}
    options=[]
    for polarity,oriented in (("original",x),("inverted",-x)):
        beats,peaks=_beats(oriented,sampling_rate)
        if len(beats)>=3:
            matrix=np.stack(beats); corr=np.corrcoef(matrix); pair=corr[np.triu_indices(len(matrix),1)]; template=np.median(matrix,axis=0); template=(template-template.mean())/max(float(template.std()),1e-8); tc=np.asarray([np.corrcoef(row,template)[0,1] for row in matrix]); rmse=np.sqrt(np.mean((matrix-template)**2,axis=1)); intervals=np.diff(peaks)/sampling_rate
            options.append({"valid":True,"reason":"ok","polarity":polarity,"n_peaks":len(peaks),"n_beats":len(matrix),"pairwise_corr_median":float(np.median(pair)),"pairwise_corr_p10":float(np.quantile(pair,0.10)),"template_corr_median":float(np.median(tc)),"template_corr_min":float(np.min(tc)),"template_nrmse_median":float(np.median(rmse)),"beat_interval_cv":float(np.std(intervals)/max(np.mean(intervals),1e-8))})
    if not options:return {"valid":False,"reason":"insufficient_complete_beats","n_beats":0}
    return max(options,key=lambda item:(int(item["n_beats"]),float(item["pairwise_corr_median"])))


def _scope(frame:pd.DataFrame,scope:str)->dict[str,float|int|str]:
    part=frame if scope=="Overall" else frame[frame.source.eq(scope)]; valid=part[part.valid]
    result={"scope":scope,"windows":len(part),"valid_windows":len(valid),"valid_percent":100*len(valid)/max(len(part),1)}
    for column in ("pairwise_corr_median","pairwise_corr_p10","template_corr_median","template_nrmse_median","n_beats"):
        values=valid[column].to_numpy(float); result.update({f"{column}_median":float(np.median(values)),f"{column}_p10":float(np.quantile(values,0.10)),f"{column}_p90":float(np.quantile(values,0.90))})
    for threshold in (0.80,0.90,0.95):result[f"pairwise_median_ge_{threshold:.2f}_percent"]=100*float((valid.pairwise_corr_median>=threshold).mean())
    subject=valid.groupby("subject_uid").pairwise_corr_median.median(); result.update({"participants":int(subject.size),"participant_median_similarity":float(subject.median()),"participant_p10_similarity":float(subject.quantile(.10)),"participant_p90_similarity":float(subject.quantile(.90))})
    return result


def analyze(store_root:Path,keys_path:Path,output:Path,*,sampling_rate:float=125.0,limit:int|None=None)->dict[str,object]:
    keys=pd.read_parquet(keys_path); required={"subject_uid","event_id","source"}
    if missing:=required-set(keys):raise ValueError(f"missing key columns: {sorted(missing)}")
    keys=keys[["subject_uid","event_id","source"]].drop_duplicates(["subject_uid","event_id"])
    if limit is not None:keys=keys.groupby("source",group_keys=False).sample(n=min(limit//max(keys.source.nunique(),1),int(keys.groupby("source").size().min())),random_state=20260820)
    metadata=load_store_metadata(store_root,"development"); selected=keys.merge(metadata[["subject_uid","event_id","waveform_file","waveform_row"]],on=["subject_uid","event_id"],validate="one_to_one"); accessor=WaveformAccessor(store_root); rows=[]
    for row in selected.itertuples(index=False):
        metrics=beat_similarity(accessor.get(str(row.waveform_file),int(row.waveform_row)).numpy().reshape(-1),sampling_rate); rows.append({"subject_uid":row.subject_uid,"event_id":row.event_id,"source":row.source,**metrics})
    result=pd.DataFrame(rows); output.mkdir(parents=True,exist_ok=False); result.to_parquet(output/"window_similarity_private.parquet",index=False); summary=pd.DataFrame([_scope(result,s) for s in ("Overall","MIMIC","VitalDB")]); summary.to_csv(output/"summary.csv",index=False)
    payload={"status":"complete","split":"meta_validation_inputs","locked_test_accessed":False,"signal":"PulseDB PPG_F representative 10-second window","sampling_rate_hz":sampling_rate,"beat_definition":"adaptive local maxima; inter-peak minima define trough-to-trough beats; each beat resampled to 100 points and z-normalized","primary_metric":"median Pearson correlation across all beat pairs within a window","descriptive_thresholds":"0.80/0.90/0.95 are descriptive only, not validated universal quality cutoffs","scopes":summary.to_dict("records")}; save_json(output/"run.json",payload)
    lines=["# 10-second PPG beat-to-beat similarity", "", "This is a morphology-consistency audit, not proof of physiological validity or motion-free acquisition.", "", "| Scope | Windows | Valid % | Pairwise median | Pairwise p10 | Template median | >=0.90 % |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['scope']} | {row['windows']} | {row['valid_percent']:.2f} | "
            f"{row['pairwise_corr_median_median']:.4f} | "
            f"{row['pairwise_corr_median_p10']:.4f} | "
            f"{row['template_corr_median_median']:.4f} | "
            f"{row['pairwise_median_ge_0.90_percent']:.2f} |"
        )
    (output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); return payload


def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--store-root",type=Path,required=True);p.add_argument("--keys",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--sampling-rate",type=float,default=125.0);p.add_argument("--limit",type=int);a=p.parse_args();print(json.dumps(analyze(a.store_root,a.keys,a.output,sampling_rate=a.sampling_rate,limit=a.limit),indent=2))


if __name__=="__main__":main()
