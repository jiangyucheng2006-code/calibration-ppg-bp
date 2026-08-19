"""Leakage-safe Phase-6E continuous residual and morphology-mixture screen."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn

from .report_phase6d_pipeline import _assert_qgate_huber_run, _load_validation_run
from .tail_risk import FEATURE_COLUMNS, build_risk_features
from .train import _load_population_checkpoint
from .training import WaveformAccessor, load_store_metadata, participant_macro_metrics, save_json, seed_everything


KEYS = ["subject_uid", "event_id", "k"]
METHODS = ("ridge", "mlp", "gated_mlp", "weighted_mlp", "causal_gru", "supervised_moe")


def prepare_validation_features(store_root: Path, general_run: Path, output: Path) -> dict[str, object]:
    predictions, record = _load_validation_run(general_run, k=5)
    _assert_qgate_huber_run(record, label="Phase-6E general")
    features = build_risk_features(store_root, predictions)
    if len(features) != len(predictions):
        raise AssertionError("validation feature and prediction counts differ")
    if features.duplicated(KEYS).any():
        raise AssertionError("validation features contain duplicate query keys")
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output, index=False)
    payload = {
        "status": "complete",
        "split": "meta_validation",
        "locked_test_accessed": False,
        "general_job": record.get("slurm_job_id"),
        "participants": int(features.subject_uid.nunique()),
        "queries": int(len(features)),
        "features": FEATURE_COLUMNS,
        "output": output.name,
    }
    save_json(output.with_suffix(".json"), payload)
    return payload


def _validate_frames(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    required = set(KEYS + FEATURE_COLUMNS + ["target_sbp", "target_dbp", "pred_sbp", "pred_dbp", "source"])
    if missing := required - set(train.columns):
        raise ValueError(f"OOF table missing {sorted(missing)}")
    if missing := required - set(validation.columns):
        raise ValueError(f"validation table missing {sorted(missing)}")
    if set(train.subject_uid.astype(str)) & set(validation.subject_uid.astype(str)):
        raise AssertionError("meta-train and meta-validation participants overlap")
    if train.split.ne("meta_train").any() or validation.split.ne("meta_validation").any():
        raise AssertionError("unexpected split in Phase-6E features")
    if train.label_split.ne("meta_train_crossfit_oof").any():
        raise AssertionError("residual training labels are not cross-fitted OOF")
    for frame in (train, validation):
        if frame.duplicated(KEYS).any():
            raise AssertionError("duplicate query keys")
        if not np.isfinite(frame[FEATURE_COLUMNS].to_numpy(float)).all():
            raise AssertionError("non-finite deployment feature")


@dataclass
class Scale:
    mean: np.ndarray
    std: np.ndarray
    residual_std: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "Scale":
        x = frame[FEATURE_COLUMNS].to_numpy(np.float32)
        residual = frame[["target_sbp", "target_dbp"]].to_numpy(np.float32) - frame[["pred_sbp", "pred_dbp"]].to_numpy(np.float32)
        mean = x.mean(0)
        std = x.std(0)
        std[std <= 1e-8] = 1.0
        residual_std = residual.std(0)
        residual_std[residual_std <= 1e-8] = 1.0
        return cls(mean, std, residual_std)

    def x(self, frame: pd.DataFrame) -> np.ndarray:
        return ((frame[FEATURE_COLUMNS].to_numpy(np.float32) - self.mean) / self.std).astype(np.float32)

    def y(self, frame: pd.DataFrame) -> np.ndarray:
        residual = frame[["target_sbp", "target_dbp"]].to_numpy(np.float32) - frame[["pred_sbp", "pred_dbp"]].to_numpy(np.float32)
        return (residual / self.residual_std).astype(np.float32)


class ResidualMLP(nn.Module):
    def __init__(self, dimension: int, *, gated: bool = False) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(dimension, 96), nn.SiLU(), nn.Dropout(0.1), nn.Linear(96, 64), nn.SiLU())
        self.delta = nn.Linear(64, 2)
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)
        self.gate = nn.Linear(64, 2) if gated else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, -1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.body(x)
        delta = self.delta(hidden)
        return delta if self.gate is None else torch.sigmoid(self.gate(hidden)) * delta


class SupervisedMoE(nn.Module):
    def __init__(self, dimension: int, experts: int = 4) -> None:
        super().__init__(); self.experts = experts
        self.body = nn.Sequential(nn.Linear(dimension, 96), nn.SiLU(), nn.Linear(96, 64), nn.SiLU())
        self.gate = nn.Linear(64, experts)
        self.heads = nn.Linear(64, experts * 2)
        nn.init.zeros_(self.heads.weight); nn.init.zeros_(self.heads.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(x)
        weights = torch.softmax(self.gate(hidden), dim=-1)
        values = self.heads(hidden).reshape(-1, self.experts, 2)
        return torch.sum(weights[..., None] * values, dim=1), weights


class CausalResidualGRU(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.gru = nn.GRU(dimension, 64, batch_first=True)
        self.head = nn.Linear(64, 2)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.gru(x)[0])


class ClusterResidualExperts(nn.Module):
    def __init__(self, dimension: int, clusters: int) -> None:
        super().__init__(); self.clusters = clusters
        self.body = nn.Sequential(nn.Linear(dimension, 96), nn.SiLU(), nn.Linear(96, 64), nn.SiLU())
        self.heads = nn.Linear(64, clusters * 2)
        nn.init.zeros_(self.heads.weight); nn.init.zeros_(self.heads.bias)

    def forward(self, x: torch.Tensor, membership: torch.Tensor) -> torch.Tensor:
        values = self.heads(self.body(x)).reshape(-1, self.clusters, 2)
        return torch.sum(membership[..., None] * values, dim=1)


def _metric(frame: pd.DataFrame, correction: np.ndarray) -> float:
    scored = frame[KEYS + ["target_sbp", "target_dbp", "pred_sbp", "pred_dbp"]].copy()
    scored[["pred_sbp", "pred_dbp"]] += correction
    return float(participant_macro_metrics(scored)["mean_mae"])


def _ridge(train: pd.DataFrame, internal: pd.DataFrame, scale: Scale) -> tuple[dict[str, object], np.ndarray]:
    x_train, y_train = scale.x(train), scale.y(train)
    x_val = scale.x(internal)
    lambdas = (0.1, 1.0, 10.0, 100.0)
    best: tuple[float, float, np.ndarray] | None = None
    xtx = x_train.T @ x_train
    xty = x_train.T @ y_train
    for value in lambdas:
        weights = np.linalg.solve(xtx + value * np.eye(xtx.shape[0], dtype=np.float32), xty)
        correction = (x_val @ weights) * scale.residual_std
        score = _metric(internal, correction)
        if best is None or score < best[0]:
            best = (score, value, weights)
    assert best is not None
    return {"internal_mean_mae": best[0], "ridge_lambda": best[1]}, best[2]


def _participant_sequences(frame: pd.DataFrame, x: np.ndarray, y: np.ndarray | None = None) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    rows: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    indexed = frame.reset_index(drop=True)
    for _, group in indexed.groupby("subject_uid", sort=True):
        order = group.sort_values(["event_index", "event_id"], kind="mergesort").index.to_numpy()
        rows.append((x[order], None if y is None else y[order], order))
    return rows


def _predict_model(model: nn.Module, method: str, x: np.ndarray, frame: pd.DataFrame, device: torch.device, membership: np.ndarray | None = None) -> np.ndarray:
    model.eval(); result = np.zeros((len(frame), 2), dtype=np.float32)
    with torch.no_grad():
        if method == "causal_gru":
            for seq, _, order in _participant_sequences(frame, x):
                result[order] = model(torch.from_numpy(seq)[None].to(device))[0].cpu().numpy()
        else:
            for start in range(0, len(frame), 4096):
                stop = min(start + 4096, len(frame)); xb = torch.from_numpy(x[start:stop]).to(device)
                if method == "supervised_moe": output = model(xb)[0]
                elif method == "cluster_moe":
                    assert membership is not None
                    output = model(xb, torch.from_numpy(membership[start:stop]).to(device))
                else: output = model(xb)
                result[start:stop] = output.cpu().numpy()
    return result


def _train_neural(method: str, train: pd.DataFrame, internal: pd.DataFrame, scale: Scale, *, seed: int, device: torch.device, train_membership: np.ndarray | None = None, val_membership: np.ndarray | None = None, clusters: int | None = None) -> tuple[nn.Module, list[dict[str, float]], int]:
    seed_everything(seed)
    if method in {"mlp", "weighted_mlp"}: model: nn.Module = ResidualMLP(len(FEATURE_COLUMNS))
    elif method == "gated_mlp": model = ResidualMLP(len(FEATURE_COLUMNS), gated=True)
    elif method == "causal_gru": model = CausalResidualGRU(len(FEATURE_COLUMNS))
    elif method == "supervised_moe": model = SupervisedMoE(len(FEATURE_COLUMNS))
    elif method == "cluster_moe":
        if clusters is None: raise ValueError("cluster count required")
        model = ClusterResidualExperts(len(FEATURE_COLUMNS), clusters)
    else: raise ValueError(method)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    x_train, y_train = scale.x(train), scale.y(train); x_val = scale.x(internal)
    weights = np.where(train.get("hard_oof", pd.Series(False, index=train.index)).to_numpy(bool), 2.0, 1.0).astype(np.float32)
    best_score, best_epoch, patience, stale = math.inf, 0, 8, 0
    best_state: dict[str, torch.Tensor] | None = None; history=[]
    sequences = _participant_sequences(train, x_train, y_train) if method == "causal_gru" else None
    for epoch in range(1, 101):
        model.train(); total=0.0; examples=0
        if sequences is not None:
            order = np.random.default_rng(seed + epoch).permutation(len(sequences))
            batches = [(sequences[i][0], sequences[i][1], None, None) for i in order]
        else:
            order = np.random.default_rng(seed + epoch).permutation(len(train)); batches=[]
            for start in range(0, len(order), 1024):
                idx=order[start:start+1024]
                batches.append((x_train[idx], y_train[idx], weights[idx], None if train_membership is None else train_membership[idx]))
        for xb_np,yb_np,wb_np,mb_np in batches:
            xb=torch.from_numpy(xb_np).to(device); yb=torch.from_numpy(yb_np).to(device)
            optimizer.zero_grad(set_to_none=True)
            if method == "causal_gru": pred=model(xb[None])[0]
            elif method == "supervised_moe":
                pred, gate=model(xb); balance=((gate.mean(0)-1.0/gate.shape[1])**2).mean()
            elif method == "cluster_moe": pred=model(xb,torch.from_numpy(mb_np).to(device))
            else: pred=model(xb)
            loss=torch.nn.functional.huber_loss(pred,yb,reduction="none",delta=0.5).mean(1)
            if method == "weighted_mlp" and wb_np is not None: loss=(loss*torch.from_numpy(wb_np).to(device)).sum()/torch.from_numpy(wb_np).to(device).sum()
            else: loss=loss.mean()
            if method == "supervised_moe": loss=loss+0.01*balance
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step()
            total += float(loss.detach())*len(yb); examples += len(yb)
        normalized = _predict_model(model, method, x_val, internal, device, val_membership)
        score = _metric(internal, normalized * scale.residual_std)
        history.append({"epoch":epoch,"train_loss":total/max(examples,1),"internal_mean_mae":score})
        if score < best_score:
            best_score=score; best_epoch=epoch; stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale += 1
        if stale >= patience: break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, history, best_epoch


def _save_predictions(validation: pd.DataFrame, correction: np.ndarray, output: Path, payload: dict[str, object]) -> dict[str, object]:
    predictions = validation[KEYS + ["target_sbp", "target_dbp", "pred_sbp", "pred_dbp", "source"]].copy()
    predictions.rename(columns={"pred_sbp":"base_pred_sbp","pred_dbp":"base_pred_dbp"}, inplace=True)
    predictions["pred_sbp"] = predictions.base_pred_sbp + correction[:,0]
    predictions["pred_dbp"] = predictions.base_pred_dbp + correction[:,1]
    metrics={}
    for scope, group in [("Overall",predictions)]+[(s,predictions[predictions.source.eq(s)]) for s in sorted(predictions.source.unique())]:
        metrics[scope]=participant_macro_metrics(group)
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(output/"predictions.parquet",index=False)
    payload.update({"status":"complete","split":"meta_validation","locked_test_accessed":False,"metrics":metrics,"participants":int(predictions.subject_uid.nunique()),"queries":int(len(predictions))})
    save_json(output/"run.json",payload)
    return payload


def train_tabular(method: str, train_path: Path, validation_path: Path, output: Path, *, seed: int) -> dict[str, object]:
    if method not in METHODS: raise ValueError(f"unsupported method {method}")
    train=pd.read_parquet(train_path); validation=pd.read_parquet(validation_path); _validate_frames(train,validation)
    fit=train[train.fold.ne(4)].reset_index(drop=True); internal=train[train.fold.eq(4)].reset_index(drop=True)
    scale=Scale.fit(fit); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload={"method":method,"seed":seed,"training_split":"meta_train_crossfit_oof_folds_0_3","selection_split":"meta_train_crossfit_fold_4","feature_columns":FEATURE_COLUMNS}
    if method=="ridge":
        selected,coef=_ridge(fit,internal,scale); correction=(scale.x(validation)@coef)*scale.residual_std
        payload.update(selected); payload["coefficients"]=coef.tolist()
    else:
        model,history,best_epoch=_train_neural(method,fit,internal,scale,seed=seed,device=device)
        correction=_predict_model(model,method,scale.x(validation),validation,device)*scale.residual_std
        payload.update({"best_epoch":best_epoch,"history":history,"model_state":None})
        output.mkdir(parents=True,exist_ok=False)
        torch.save({"model_state":model.state_dict(),"scale":{"mean":scale.mean,"std":scale.std,"residual_std":scale.residual_std},"method":method,"seed":seed},output/"best.pt")
        output.rmdir() if False else None
        # _save_predictions expects to create the directory; it already exists for the checkpoint.
        predictions=validation[KEYS+["target_sbp","target_dbp","pred_sbp","pred_dbp","source"]].copy()
        predictions.rename(columns={"pred_sbp":"base_pred_sbp","pred_dbp":"base_pred_dbp"},inplace=True)
        predictions["pred_sbp"]=predictions.base_pred_sbp+correction[:,0]; predictions["pred_dbp"]=predictions.base_pred_dbp+correction[:,1]
        metrics={scope:participant_macro_metrics(group) for scope,group in [("Overall",predictions)]+[(s,predictions[predictions.source.eq(s)]) for s in sorted(predictions.source.unique())]}
        predictions.to_parquet(output/"predictions.parquet",index=False); payload.update({"status":"complete","split":"meta_validation","locked_test_accessed":False,"metrics":metrics,"participants":int(predictions.subject_uid.nunique()),"queries":int(len(predictions))}); save_json(output/"run.json",payload); return payload
    return _save_predictions(validation,correction,output,payload)


def extract_embeddings(store_root: Path, population_checkpoint: Path, train_path: Path, validation_path: Path, output: Path, *, batch_size: int=1024) -> dict[str, object]:
    train=pd.read_parquet(train_path); validation=pd.read_parquet(validation_path); _validate_frames(train,validation)
    wanted=pd.concat([train[KEYS],validation[KEYS]],ignore_index=True).drop_duplicates(KEYS)
    metadata=load_store_metadata(store_root,"development")
    selected=metadata.merge(wanted[["subject_uid","event_id"]].drop_duplicates(),on=["subject_uid","event_id"],how="inner",validate="one_to_one").sort_values(["subject_uid","event_id"]).reset_index(drop=True)
    if len(selected)!=len(wanted): raise AssertionError("embedding metadata does not cover query keys")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); population,_=_load_population_checkpoint(population_checkpoint,device); population.to(device).eval(); accessor=WaveformAccessor(store_root)
    values=[]
    with torch.no_grad():
        for start in range(0,len(selected),batch_size):
            part=selected.iloc[start:start+batch_size]
            batch=torch.stack([accessor.get(str(r.waveform_file),int(r.waveform_row)) for r in part.itertuples()]).to(device)
            values.append(population.encoder(batch).cpu().numpy().astype(np.float32))
    embedding=np.concatenate(values); columns=[f"embedding_{i:03d}" for i in range(embedding.shape[1])]
    result=selected[["subject_uid","event_id","split","source"]].copy()
    result=pd.concat([result,pd.DataFrame(embedding,columns=columns)],axis=1)
    output.parent.mkdir(parents=True,exist_ok=True); result.to_parquet(output,index=False)
    payload={"status":"complete","locked_test_accessed":False,"events":len(result),"dimension":embedding.shape[1],"population_checkpoint_sha256":None,"output":output.name}; save_json(output.with_suffix(".json"),payload); return payload


def _kmeans(x: np.ndarray, k: int, seed: int, iterations: int=15) -> tuple[np.ndarray,np.ndarray,float]:
    rng=np.random.default_rng(seed); centroids=x[rng.choice(len(x),k,replace=False)].copy()
    for _ in range(iterations):
        distance=((x[:,None,:]-centroids[None,:,:])**2).sum(2); labels=distance.argmin(1); new=[]
        for cluster in range(k):
            members=x[labels==cluster]; new.append(members.mean(0) if len(members) else x[rng.integers(len(x))])
        updated=np.asarray(new,np.float32)
        if np.allclose(updated,centroids,rtol=1e-4,atol=1e-4): centroids=updated; break
        centroids=updated
    distance=((x[:,None,:]-centroids[None,:,:])**2).sum(2); labels=distance.argmin(1)
    return centroids,labels,float(distance[np.arange(len(x)),labels].mean())


def _cluster_stability(x: np.ndarray,k: int,seed:int) -> tuple[np.ndarray,dict[str,float]]:
    a,la,inertia=_kmeans(x,k,seed); b,lb,_=_kmeans(x,k,seed+1)
    mapping={}; available=set(range(k))
    for i in range(k):
        j=min(available,key=lambda q:float(((a[i]-b[q])**2).sum())); mapping[j]=i; available.remove(j)
    remapped=np.asarray([mapping[int(j)] for j in lb]); counts=np.bincount(la,minlength=k)
    return a,{"k":k,"stability":float((la==remapped).mean()),"minimum_cluster_fraction":float(counts.min()/len(x)),"inertia":inertia}


def _source_mixing_entropy(labels: np.ndarray, sources: np.ndarray, k: int) -> float:
    """Return participant-source mixing entropy in [0,1] for audit only."""
    values=[]; weights=[]
    for cluster in range(k):
        selected=sources[labels==cluster]
        if len(selected)==0: continue
        proportions=pd.Series(selected).value_counts(normalize=True).to_numpy(float)
        entropy=-float(np.sum(proportions*np.log(proportions+1e-12)))
        maximum=math.log(max(len(np.unique(sources)),2))
        values.append(min(1.0,max(0.0,entropy/maximum))); weights.append(len(selected))
    return float(np.average(values,weights=weights))


def _select_cluster_count(audits: list[dict[str, float]]) -> tuple[int, bool]:
    passing=[a for a in audits if a["stability"]>=0.75 and a["minimum_cluster_fraction"]>=0.005]
    return max((int(a["k"]) for a in passing),default=8),bool(passing)


def train_cluster_moe(train_path: Path, validation_path: Path, embedding_path: Path, output: Path, *, seed:int) -> dict[str,object]:
    train=pd.read_parquet(train_path); validation=pd.read_parquet(validation_path); _validate_frames(train,validation); embeddings=pd.read_parquet(embedding_path)
    ecols=[c for c in embeddings if c.startswith("embedding_")]
    train=train.merge(embeddings[["subject_uid","event_id"]+ecols],on=["subject_uid","event_id"],validate="one_to_one"); validation=validation.merge(embeddings[["subject_uid","event_id"]+ecols],on=["subject_uid","event_id"],validate="one_to_one")
    fit=train[train.fold.ne(4)].reset_index(drop=True); internal=train[train.fold.eq(4)].reset_index(drop=True)
    sample=fit.sample(n=min(20000,len(fit)),random_state=seed); raw=sample[ecols].to_numpy(np.float32); center=raw.mean(0); centered=raw-center
    _,_,v=np.linalg.svd(centered[:min(20000,len(centered))],full_matrices=False); projection=v[:32].T.astype(np.float32)
    projected=((raw-center)@projection).astype(np.float32); pmean=projected.mean(0); pstd=projected.std(0); pstd[pstd<=1e-8]=1; projected=(projected-pmean)/pstd
    audits=[]; candidates={}
    for k in (8,16,32):
        centroids,audit=_cluster_stability(projected,k,seed+k)
        assignments=((projected[:,None,:]-centroids[None,:,:])**2).sum(2).argmin(1)
        audit["source_mixing_entropy"]=_source_mixing_entropy(assignments,sample.source.to_numpy(),k)
        audits.append(audit); candidates[k]=centroids
    selected_k,selected_by_gate=_select_cluster_count(audits); centroids=candidates[selected_k]
    def memberships(frame:pd.DataFrame)->np.ndarray:
        z=((frame[ecols].to_numpy(np.float32)-center)@projection-pmean)/pstd
        distance=((z[:,None,:]-centroids[None,:,:])**2).sum(2); temperature=max(float(np.median(distance.min(1))),1e-4)
        logits=-distance/temperature; logits-=logits.max(1,keepdims=True); weights=np.exp(logits); return (weights/weights.sum(1,keepdims=True)).astype(np.float32)
    fit_membership=memberships(fit); internal_membership=memberships(internal); validation_membership=memberships(validation)
    scale=Scale.fit(fit); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,history,best_epoch=_train_neural("cluster_moe",fit,internal,scale,seed=seed,device=device,train_membership=fit_membership,val_membership=internal_membership,clusters=selected_k)
    correction=_predict_model(model,"cluster_moe",scale.x(validation),validation,device,validation_membership)*scale.residual_std
    output.mkdir(parents=True,exist_ok=False); torch.save({"model_state":model.state_dict(),"selected_k":selected_k,"cluster_audit":audits,"projection":projection,"embedding_center":center,"projected_mean":pmean,"projected_std":pstd,"centroids":centroids,"feature_mean":scale.mean,"feature_std":scale.std,"residual_std":scale.residual_std},output/"best.pt")
    predictions=validation[KEYS+["target_sbp","target_dbp","pred_sbp","pred_dbp","source"]].copy(); predictions.rename(columns={"pred_sbp":"base_pred_sbp","pred_dbp":"base_pred_dbp"},inplace=True); predictions["pred_sbp"]=predictions.base_pred_sbp+correction[:,0]; predictions["pred_dbp"]=predictions.base_pred_dbp+correction[:,1]; predictions["cluster_top1"]=validation_membership.argmax(1); predictions["cluster_confidence"]=validation_membership.max(1); predictions.to_parquet(output/"predictions.parquet",index=False)
    metrics={scope:participant_macro_metrics(group) for scope,group in [("Overall",predictions)]+[(s,predictions[predictions.source.eq(s)]) for s in sorted(predictions.source.unique())]}; payload={"status":"complete","method":"morphology_cluster_moe","seed":seed,"split":"meta_validation","locked_test_accessed":False,"selected_k":selected_k,"selected_by_stability_gate":selected_by_gate,"cluster_selection_status":"passed_meta_train_gate" if selected_by_gate else "exploratory_fallback_k8_no_candidate_passed","cluster_selection":"highest K passing meta-train-only stability>=0.75 and min cluster fraction>=0.005; if none pass, K=8 is retained only as an explicitly exploratory fallback and is ineligible for promotion","cluster_audit":audits,"best_epoch":best_epoch,"history":history,"metrics":metrics,"participants":int(predictions.subject_uid.nunique()),"queries":len(predictions)}; save_json(output/"run.json",payload); return payload


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command",required=True)
    prepare=sub.add_parser("prepare-validation"); prepare.add_argument("--store-root",type=Path,required=True); prepare.add_argument("--general-run",type=Path,required=True); prepare.add_argument("--output",type=Path,required=True)
    train=sub.add_parser("train"); train.add_argument("--method",choices=METHODS,required=True); train.add_argument("--train-features",type=Path,required=True); train.add_argument("--validation-features",type=Path,required=True); train.add_argument("--output",type=Path,required=True); train.add_argument("--seed",type=int,default=20260820)
    embed=sub.add_parser("extract-embeddings"); embed.add_argument("--store-root",type=Path,required=True); embed.add_argument("--population-checkpoint",type=Path,required=True); embed.add_argument("--train-features",type=Path,required=True); embed.add_argument("--validation-features",type=Path,required=True); embed.add_argument("--output",type=Path,required=True)
    cluster=sub.add_parser("train-cluster-moe"); cluster.add_argument("--train-features",type=Path,required=True); cluster.add_argument("--validation-features",type=Path,required=True); cluster.add_argument("--embeddings",type=Path,required=True); cluster.add_argument("--output",type=Path,required=True); cluster.add_argument("--seed",type=int,default=20260820)
    args=parser.parse_args()
    if args.command=="prepare-validation": payload=prepare_validation_features(args.store_root,args.general_run,args.output)
    elif args.command=="train": payload=train_tabular(args.method,args.train_features,args.validation_features,args.output,seed=args.seed)
    elif args.command=="extract-embeddings": payload=extract_embeddings(args.store_root,args.population_checkpoint,args.train_features,args.validation_features,args.output)
    else: payload=train_cluster_moe(args.train_features,args.validation_features,args.embeddings,args.output,seed=args.seed)
    print(json.dumps(payload,indent=2,default=str))


if __name__=="__main__": main()
