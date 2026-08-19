"""Round-7 waveform phenotype routing and deep temporal residual models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .phase6e_residual import (
    FEATURE_COLUMNS,
    KEYS,
    ResidualMLP,
    Scale,
    _cluster_stability,
    _predict_model,
    _source_mixing_entropy,
    _train_neural,
    _validate_frames,
)
from .training import participant_macro_metrics, save_json, seed_everything


class EmbeddingAutoencoder(nn.Module):
    def __init__(self, dimension: int, latent: int = 32) -> None:
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(dimension,128),nn.SiLU(),nn.Linear(128,latent))
        self.decoder=nn.Sequential(nn.Linear(latent,128),nn.SiLU(),nn.Linear(128,dimension))

    def forward(self,x:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
        z=self.encoder(x); return self.decoder(z),z


class PhenotypeRouter(nn.Module):
    def __init__(self,dimension:int,clusters:int)->None:
        super().__init__(); self.net=nn.Sequential(nn.Linear(dimension,64),nn.SiLU(),nn.Linear(64,clusters))
    def forward(self,x:torch.Tensor)->torch.Tensor: return self.net(x)


class EmbeddingCausalGRU(nn.Module):
    def __init__(self,dimension:int)->None:
        super().__init__(); self.gru=nn.GRU(dimension,96,batch_first=True); self.head=nn.Linear(96,2)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self,x:torch.Tensor)->torch.Tensor: return self.head(self.gru(x)[0])


def _load(train_path:Path,validation_path:Path,embedding_path:Path)->tuple[pd.DataFrame,pd.DataFrame,list[str]]:
    train=pd.read_parquet(train_path); validation=pd.read_parquet(validation_path); _validate_frames(train,validation)
    embeddings=pd.read_parquet(embedding_path); ecols=[c for c in embeddings if c.startswith("embedding_")]
    cols=["subject_uid","event_id"]+ecols
    train=train.merge(embeddings[cols],on=["subject_uid","event_id"],validate="one_to_one")
    validation=validation.merge(embeddings[cols],on=["subject_uid","event_id"],validate="one_to_one")
    return train,validation,ecols


def _encode(model:EmbeddingAutoencoder,x:np.ndarray,device:torch.device)->np.ndarray:
    model.eval(); out=[]
    with torch.no_grad():
        for start in range(0,len(x),4096): out.append(model.encoder(torch.from_numpy(x[start:start+4096]).to(device)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def train_router(train_path:Path,validation_path:Path,embedding_path:Path,output:Path,*,seed:int,clusters:int)->dict[str,object]:
    train,validation,ecols=_load(train_path,validation_path,embedding_path); fit=train[train.fold.ne(4)].reset_index(drop=True); internal=train[train.fold.eq(4)].reset_index(drop=True)
    seed_everything(seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean=fit[ecols].to_numpy(np.float32).mean(0); std=fit[ecols].to_numpy(np.float32).std(0); std[std<=1e-8]=1
    def norm(frame:pd.DataFrame)->np.ndarray:return ((frame[ecols].to_numpy(np.float32)-mean)/std).astype(np.float32)
    model=EmbeddingAutoencoder(len(ecols)).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4)
    xfit=norm(fit); xint=norm(internal); rng=np.random.default_rng(seed); train_idx=rng.choice(len(xfit),min(120000,len(xfit)),replace=False); val_idx=rng.choice(len(xint),min(30000,len(xint)),replace=False)
    best=math.inf; best_state=None; stale=0; history=[]
    for epoch in range(1,41):
        model.train(); order=rng.permutation(train_idx); total=0.0
        for start in range(0,len(order),2048):
            xb=torch.from_numpy(xfit[order[start:start+2048]]).to(device); optimizer.zero_grad(set_to_none=True); recon,_=model(xb); loss=torch.nn.functional.mse_loss(recon,xb); loss.backward(); optimizer.step(); total+=float(loss)*len(xb)
        model.eval()
        with torch.no_grad(): score=float(torch.nn.functional.mse_loss(model(torch.from_numpy(xint[val_idx]).to(device))[0],torch.from_numpy(xint[val_idx]).to(device)))
        history.append({"epoch":epoch,"train_reconstruction":total/len(order),"internal_reconstruction":score})
        if score<best: best=score; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=6: break
    assert best_state is not None; model.load_state_dict(best_state)
    zfit=_encode(model,xfit,device); zint=_encode(model,xint,device); zval=_encode(model,norm(validation),device)
    sample_idx=rng.choice(len(zfit),min(30000,len(zfit)),replace=False); centroids,audit=_cluster_stability(zfit[sample_idx],clusters,seed+clusters)
    fit_label=((zfit[:,None,:]-centroids[None,:,:])**2).sum(2).argmin(1); int_label=((zint[:,None,:]-centroids[None,:,:])**2).sum(2).argmin(1)
    audit["source_mixing_entropy"]=_source_mixing_entropy(fit_label,fit.source.to_numpy(),clusters)
    router=PhenotypeRouter(zfit.shape[1],clusters).to(device); opt=torch.optim.AdamW(router.parameters(),lr=1e-3,weight_decay=1e-4); best_acc=-1.0; router_state=None; router_history=[]
    for epoch in range(1,31):
        router.train(); order=rng.permutation(len(zfit)); total=0.0
        for start in range(0,len(order),2048):
            idx=order[start:start+2048]; xb=torch.from_numpy(zfit[idx]).to(device); yb=torch.from_numpy(fit_label[idx]).long().to(device); opt.zero_grad(set_to_none=True); loss=torch.nn.functional.cross_entropy(router(xb),yb); loss.backward(); opt.step(); total+=float(loss)*len(idx)
        router.eval()
        with torch.no_grad(): acc=float((router(torch.from_numpy(zint).to(device)).argmax(1).cpu().numpy()==int_label).mean())
        router_history.append({"epoch":epoch,"train_loss":total/len(order),"internal_pseudo_label_accuracy":acc})
        if acc>best_acc: best_acc=acc; router_state={k:v.detach().cpu().clone() for k,v in router.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=6: break
    assert router_state is not None; router.load_state_dict(router_state); router.eval()
    def route(z:np.ndarray)->tuple[np.ndarray,np.ndarray]:
        out=[]
        with torch.no_grad():
            for start in range(0,len(z),4096): out.append(torch.softmax(router(torch.from_numpy(z[start:start+4096]).to(device)),1).cpu().numpy())
        probability=np.concatenate(out).astype(np.float32); return probability.argmax(1),probability
    frames=[]
    for frame,z,name in ((fit,zfit,"fit"),(internal,zint,"internal"),(validation,zval,"validation")):
        label,prob=route(z); item=frame[KEYS+["split","source"]+(["fold"] if "fold" in frame else [])].copy(); item["router_role"]=name; item["phenotype"]=label
        for i in range(z.shape[1]): item[f"latent_{i:02d}"]=z[:,i]
        for i in range(clusters): item[f"phenotype_probability_{i:02d}"]=prob[:,i]
        frames.append(item)
    output.mkdir(parents=True,exist_ok=False); routed=pd.concat(frames,ignore_index=True); routed.to_parquet(output/"routed_features.parquet",index=False)
    torch.save({"autoencoder":model.state_dict(),"router":router.state_dict(),"embedding_mean":mean,"embedding_std":std,"centroids":centroids,"clusters":clusters},output/"router.pt")
    payload={"status":"complete","method":"two_stage_waveform_phenotype_router","seed":seed,"clusters":clusters,"training_split":"meta_train_crossfit_oof_folds_0_3","selection_split":"meta_train_crossfit_fold_4","split":"meta_train_only_router","locked_test_accessed":False,"autoencoder_history":history,"router_history":router_history,"internal_router_accuracy":best_acc,"cluster_audit":audit,"fit_cluster_counts":pd.Series(fit_label).value_counts().sort_index().to_dict()}; save_json(output/"run.json",payload); return payload


def _finalize(validation:pd.DataFrame,correction:np.ndarray,extra:pd.DataFrame,output:Path,payload:dict[str,object])->dict[str,object]:
    pred=validation[KEYS+["target_sbp","target_dbp","pred_sbp","pred_dbp","source"]].copy(); pred.rename(columns={"pred_sbp":"base_pred_sbp","pred_dbp":"base_pred_dbp"},inplace=True); pred["pred_sbp"]=pred.base_pred_sbp+correction[:,0]; pred["pred_dbp"]=pred.base_pred_dbp+correction[:,1]; pred=pred.merge(extra,on=KEYS,validate="one_to_one")
    pred.to_parquet(output/"predictions.parquet",index=False); metrics={scope:participant_macro_metrics(group) for scope,group in [("Overall",pred)]+[(s,pred[pred.source.eq(s)]) for s in sorted(pred.source.unique())]}; payload.update({"status":"complete","split":"meta_validation","locked_test_accessed":False,"metrics":metrics,"participants":int(pred.subject_uid.nunique()),"queries":len(pred)}); save_json(output/"run.json",payload); return payload


def train_experts(train_path:Path,validation_path:Path,routed_path:Path,output:Path,*,seed:int,routing:str)->dict[str,object]:
    train=pd.read_parquet(train_path); validation=pd.read_parquet(validation_path); _validate_frames(train,validation); routed=pd.read_parquet(routed_path); probability=[c for c in routed if c.startswith("phenotype_probability_")]; clusters=len(probability); route_columns=KEYS+["phenotype"]+probability; merged=train.merge(routed[routed.split.eq("meta_train")][route_columns],on=KEYS,validate="one_to_one"); val=validation.merge(routed[routed.split.eq("meta_validation")][route_columns],on=KEYS,validate="one_to_one")
    fit=merged[merged.fold.ne(4)].reset_index(drop=True); internal=merged[merged.fold.eq(4)].reset_index(drop=True); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); correction=np.zeros((len(val),2),np.float32); expert_predictions=[]; records=[]; output.mkdir(parents=True,exist_ok=False)
    for cluster in range(clusters):
        f=fit[fit.phenotype.eq(cluster)].reset_index(drop=True); iv=internal[internal.phenotype.eq(cluster)].reset_index(drop=True)
        if len(f)<1000 or len(iv)<100: raise AssertionError(f"phenotype {cluster} is too small for an independent expert")
        scale=Scale.fit(f); model,history,best_epoch=_train_neural("mlp",f,iv,scale,seed=seed+cluster,device=device); all_pred=_predict_model(model,"mlp",scale.x(val),val,device)*scale.residual_std; expert_predictions.append(all_pred); torch.save({"model_state":model.state_dict(),"scale":scale,"phenotype":cluster},output/f"expert_{cluster:02d}.pt"); records.append({"phenotype":cluster,"fit_events":len(f),"internal_events":len(iv),"best_epoch":best_epoch,"history":history})
    stack=np.stack(expert_predictions,1); probs=val[probability].to_numpy(np.float32)
    if routing=="hard": correction=stack[np.arange(len(val)),val.phenotype.to_numpy(int)]
    else: correction=(stack*probs[...,None]).sum(1)
    extra=val[KEYS+["phenotype"]].copy(); extra["phenotype_confidence"]=probs.max(1); payload={"method":f"two_stage_phenotype_{routing}_experts","seed":seed,"clusters":clusters,"routing":routing,"experts":records,"training_split":"meta_train_crossfit_oof_folds_0_3","selection_split":"meta_train_crossfit_fold_4"}; return _finalize(val,correction,extra,output,payload)


def _sequence_order(frame:pd.DataFrame)->list[np.ndarray]:
    indexed=frame.reset_index(drop=True); return [g.sort_values(["event_index","event_id"],kind="mergesort").index.to_numpy() for _,g in indexed.groupby("subject_uid",sort=True)]


def _predict_gru(model:EmbeddingCausalGRU,x:np.ndarray,frame:pd.DataFrame,device:torch.device)->np.ndarray:
    result=np.zeros((len(frame),2),np.float32); model.eval()
    with torch.no_grad():
        for idx in _sequence_order(frame): result[idx]=model(torch.from_numpy(x[idx])[None].to(device))[0].cpu().numpy()
    return result


def train_embedding_gru(train_path:Path,validation_path:Path,routed_path:Path,output:Path,*,seed:int)->dict[str,object]:
    train=pd.read_parquet(train_path); validation=pd.read_parquet(validation_path); _validate_frames(train,validation); routed=pd.read_parquet(routed_path); latent=[c for c in routed if c.startswith("latent_")]
    train=train.merge(routed[routed.split.eq("meta_train")][KEYS+latent],on=KEYS,validate="one_to_one"); validation=validation.merge(routed[routed.split.eq("meta_validation")][KEYS+latent],on=KEYS,validate="one_to_one"); fit=train[train.fold.ne(4)].reset_index(drop=True); internal=train[train.fold.eq(4)].reset_index(drop=True)
    scale=Scale.fit(fit); lmean=fit[latent].to_numpy(np.float32).mean(0); lstd=fit[latent].to_numpy(np.float32).std(0); lstd[lstd<=1e-8]=1
    def x(frame:pd.DataFrame)->np.ndarray:return np.concatenate([scale.x(frame),(frame[latent].to_numpy(np.float32)-lmean)/lstd],1).astype(np.float32)
    def y(frame:pd.DataFrame)->np.ndarray:return scale.y(frame)
    xfit,yfit=x(fit),y(fit); xint=x(internal); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); seed_everything(seed); model=EmbeddingCausalGRU(xfit.shape[1]).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4); rng=np.random.default_rng(seed)
    chunks=[]
    for idx in _sequence_order(fit):
        for start in range(0,len(idx),32): chunks.append(idx[start:start+32])
    best=math.inf; state=None; stale=0; history=[]
    for epoch in range(1,61):
        model.train(); order=rng.permutation(len(chunks)); total=0.0; count=0
        for start in range(0,len(order),64):
            selected=[chunks[i] for i in order[start:start+64]]; length=max(map(len,selected)); xb=np.zeros((len(selected),length,xfit.shape[1]),np.float32); yb=np.zeros((len(selected),length,2),np.float32); mask=np.zeros((len(selected),length),np.float32)
            for row,idx in enumerate(selected): xb[row,:len(idx)]=xfit[idx]; yb[row,:len(idx)]=yfit[idx]; mask[row,:len(idx)]=1
            xt=torch.from_numpy(xb).to(device); yt=torch.from_numpy(yb).to(device); mt=torch.from_numpy(mask).to(device); optimizer.zero_grad(set_to_none=True); pred=model(xt); loss=(torch.nn.functional.huber_loss(pred,yt,reduction="none",delta=0.5).mean(2)*mt).sum()/mt.sum(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step(); total+=float(loss)*float(mt.sum()); count+=int(mt.sum())
        correction=_predict_gru(model,xint,internal,device)*scale.residual_std; score=float(participant_macro_metrics(internal.assign(pred_sbp=internal.pred_sbp+correction[:,0],pred_dbp=internal.pred_dbp+correction[:,1]))["mean_mae"]); history.append({"epoch":epoch,"train_loss":total/count,"internal_mean_mae":score})
        if score<best: best=score; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=8: break
    assert state is not None; model.load_state_dict(state); correction=_predict_gru(model,x(validation),validation,device)*scale.residual_std; output.mkdir(parents=True,exist_ok=False); torch.save({"model_state":state,"feature_scale":scale,"latent_mean":lmean,"latent_std":lstd},output/"best.pt"); payload={"method":"waveform_embedding_causal_gru","seed":seed,"history":history,"training_split":"meta_train_crossfit_oof_folds_0_3","selection_split":"meta_train_crossfit_fold_4"}; return _finalize(validation,correction,validation[KEYS].copy(),output,payload)


def main()->None:
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    router=sub.add_parser("train-router"); router.add_argument("--train-features",type=Path,required=True); router.add_argument("--validation-features",type=Path,required=True); router.add_argument("--embeddings",type=Path,required=True); router.add_argument("--output",type=Path,required=True); router.add_argument("--seed",type=int,default=20260821); router.add_argument("--clusters",type=int,default=8)
    expert=sub.add_parser("train-experts"); expert.add_argument("--train-features",type=Path,required=True); expert.add_argument("--validation-features",type=Path,required=True); expert.add_argument("--routed-features",type=Path,required=True); expert.add_argument("--output",type=Path,required=True); expert.add_argument("--seed",type=int,default=20260821); expert.add_argument("--routing",choices=["hard","soft"],required=True)
    temporal=sub.add_parser("train-embedding-gru"); temporal.add_argument("--train-features",type=Path,required=True); temporal.add_argument("--validation-features",type=Path,required=True); temporal.add_argument("--routed-features",type=Path,required=True); temporal.add_argument("--output",type=Path,required=True); temporal.add_argument("--seed",type=int,default=20260821)
    args=p.parse_args()
    if args.command=="train-router": result=train_router(args.train_features,args.validation_features,args.embeddings,args.output,seed=args.seed,clusters=args.clusters)
    elif args.command=="train-experts": result=train_experts(args.train_features,args.validation_features,args.routed_features,args.output,seed=args.seed,routing=args.routing)
    else: result=train_embedding_gru(args.train_features,args.validation_features,args.routed_features,args.output,seed=args.seed)
    print(json.dumps(result,indent=2,default=str))


if __name__=="__main__": main()
