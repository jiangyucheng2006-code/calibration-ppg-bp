"""Aggregate and gate Phase-6E residual-learning candidates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .report_phase6b_bootstrap import _paired_bootstrap
from .report_phase6d_pipeline import _assert_aligned, _load_validation_run, _pooled_rows, _prediction_metric_rows
from .training import save_json


KEYS=["subject_uid","event_id","k"]


def _parse(values:list[str])->dict[str,Path]:
    result={}
    for value in values:
        if "=" not in value: raise ValueError("run must be Setting=/path")
        setting,path=value.split("=",1)
        if not setting.strip() or setting in result: raise ValueError("empty or duplicate setting")
        result[setting.strip()]=Path(path)
    if not result: raise ValueError("at least one run is required")
    return result


def _participant(frame:pd.DataFrame)->pd.DataFrame:
    work=frame.copy(); work["ae_sbp"]=(work.pred_sbp-work.target_sbp).abs(); work["ae_dbp"]=(work.pred_dbp-work.target_dbp).abs()
    result=work.groupby(["source","subject_uid"],as_index=False).agg(sbp=("ae_sbp","mean"),dbp=("ae_dbp","mean")); result["mean"]=(result.sbp+result.dbp)/2; return result


def report(general_run:Path,runs:Mapping[str,Path],output:Path,*,bootstrap_seed:int=20260820)->dict[str,object]:
    general,record=_load_validation_run(general_run,k=5)
    frames={"General QGate + Huber":general}
    for setting,path in runs.items():
        run=json.loads((path/"run.json").read_text())
        if run.get("status")!="complete" or run.get("split")!="meta_validation" or run.get("locked_test_accessed") is not False: raise AssertionError(f"invalid run {setting}")
        candidate=pd.read_parquet(path/"predictions.parquet").sort_values(KEYS).reset_index(drop=True)
        _assert_aligned(general,candidate); frames[setting]=candidate
    # Candidates already contain source; restore it to the general frame from the first candidate.
    first=next(iter(runs)); source=frames[first][KEYS+["source"]]
    frames["General QGate + Huber"]=general.merge(source,on=KEYS,validate="one_to_one")
    macro=[]; pooled=[]; bootstrap=[]
    reference_participant=_participant(frames["General QGate + Huber"])
    for index,(setting,frame) in enumerate(frames.items()):
        macro.extend(_prediction_metric_rows(setting,frame)); pooled.extend(_pooled_rows(setting,frame))
        if setting=="General QGate + Huber": continue
        candidate_participant=_participant(frame)
        for scope_index,scope in enumerate(["Overall","MIMIC","VitalDB"]):
            a=candidate_participant if scope=="Overall" else candidate_participant[candidate_participant.source.eq(scope)]
            b=reference_participant if scope=="Overall" else reference_participant[reference_participant.source.eq(scope)]
            paired=a[["subject_uid","mean"]].rename(columns={"mean":"candidate"}).merge(b[["subject_uid","mean"]].rename(columns={"mean":"reference"}),on="subject_uid",validate="one_to_one")
            metric=_paired_bootstrap((paired.candidate-paired.reference).to_numpy(),repetitions=20000,seed=bootstrap_seed+index*10+scope_index)
            bootstrap.append({"setting":setting,"scope":scope,"participants":len(paired),**metric})
    macro_frame=pd.DataFrame(macro); pooled_frame=pd.DataFrame(pooled); bootstrap_frame=pd.DataFrame(bootstrap)
    overall=macro_frame[macro_frame.scope.eq("Overall")].sort_values("mean_mae"); winner=overall.iloc[0]
    reference=float(overall.loc[overall.setting.eq("General QGate + Huber"),"mean_mae"].iloc[0]); gain=reference-float(winner.mean_mae)
    source_winner=macro_frame[macro_frame.setting.eq(winner.setting)].set_index("scope")
    source_reference=macro_frame[macro_frame.setting.eq("General QGate + Huber")].set_index("scope")
    passes=bool(gain>=0.15 and all(float(source_winner.loc[s,"mean_mae"])<float(source_reference.loc[s,"mean_mae"]) for s in ("MIMIC","VitalDB")))
    output.mkdir(parents=True,exist_ok=False); macro_frame.to_csv(output/"participant_macro.csv",index=False); pooled_frame.to_csv(output/"pooled_metrics.csv",index=False); bootstrap_frame.to_csv(output/"paired_bootstrap.csv",index=False)
    payload={"status":"complete","generated_at_utc":datetime.now(timezone.utc).isoformat(),"split":"meta_validation","locked_test_accessed":False,"k":5,"general_job":record.get("slurm_job_id"),"settings":list(frames),"winner":str(winner.setting),"winner_overall_mean_mae":float(winner.mean_mae),"gain_vs_general_mmHg":gain,"promotion_threshold_mmHg":0.15,"passes_screening_gate":passes,"claim_limit":"single-seed development screening; locked test not accessed"}; save_json(output/"run.json",payload); return payload


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--general-run",type=Path,required=True); parser.add_argument("--run",action="append",default=[],required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--bootstrap-seed",type=int,default=20260820); args=parser.parse_args(); print(json.dumps(report(args.general_run,_parse(args.run),args.output,bootstrap_seed=args.bootstrap_seed),indent=2))


if __name__=="__main__": main()
