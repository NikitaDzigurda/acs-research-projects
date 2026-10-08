"""Model functions from the original ACS experiment, with local run context.

Numerical preprocessing, class handling, selection and search spaces are preserved.
Call configure(context) once per run before training. No patient data are embedded.
"""
import numpy as np, pandas as pd, json, datetime
import time, warnings, joblib, optuna, sqlite3, math
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, confusion_matrix, roc_curve, precision_score, recall_score, f1_score, accuracy_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from threadpoolctl import threadpool_limits
threadpool_limits(2)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def configure(context):
    required = ["meta", "BASE_SEED", "BASE", "FAMILIES", "STRATEGIES", "CONFIG",
                "RUN_ID", "STORAGE", "D", "groups", "y", "OUT", "SAMPLING_AUDIT", "log"]
    globals().update({name: context[name] for name in required})

def make_preprocessor(columns,family,indicators):
    nums=[c for c in columns if c in meta["continuous"]];cats=[c for c in columns if c not in nums]
    numeric_steps=[("impute",SimpleImputer(strategy="median",keep_empty_features=True))]
    if family=="LR":numeric_steps.append(("scale",StandardScaler()))
    categories=[[1,2,3,4] if c=="killip" else [0,1] for c in cats]
    transformers=[("numeric",Pipeline(numeric_steps),nums),("categorical",Pipeline([("impute",SimpleImputer(strategy="most_frequent",keep_empty_features=True)),("onehot",OneHotEncoder(categories=categories,handle_unknown="ignore",sparse_output=False))]),cats)]
    indicator_columns=[c for c in columns if c != "crp"]
    if indicators:transformers.append(("missing",MissingIndicator(features="all",error_on_new=False),indicator_columns))
    return ColumnTransformer(transformers,verbose_feature_names_out=True,sparse_threshold=0)
def sample_indices(a,b,strategy,seed):
    rng=np.random.default_rng(seed);dead=np.flatnonzero(b==1);alive=np.flatnonzero(b==0);target=min(4*len(dead),len(alive));median=float(a.age.median())
    allocations={}
    if strategy in ["all_real","class_weights"]: ids=np.arange(len(b))
    elif strategy=="random_4to1":ids=np.r_[dead,rng.choice(alive,target,replace=False)]
    else:
        age=np.where(a.age.isna(),"missing",np.where(a.age.to_numpy()>median,"older","younger"))
        killip=np.where(a.killip.isna(),"missing",np.where(a.killip.to_numpy()>=3,"3-4","1-2"))
        keys=np.array([str(k)+"|"+str(v) for k,v in zip(killip,age)])
        names,counts=np.unique(keys[alive],return_counts=True)
        quotas=target*counts/len(alive);take=np.floor(quotas).astype(int)
        order=np.argsort(-(quotas-take),kind="stable")
        for i in order[:target-int(take.sum())]:take[i]+=1
        assert take.sum()==target and np.all(take<=counts)
        chosen=[rng.choice(alive[keys[alive]==key],int(k),replace=False) for key,k in zip(names,take)]
        ids=np.r_[dead,*chosen];allocations={str(k):dict(available=int(n),selected=int(t)) for k,n,t in zip(names,counts,take)}
    assert set(dead).issubset(ids) and len(ids)==len(np.unique(ids))
    if "4to1" in strategy:assert sum(b[ids]==0)==target
    SAMPLING_AUDIT.append(dict(seed=seed,strategy=strategy,train_n=len(b),train_deaths=len(dead),selected_n=len(ids),selected_deaths=int(b[ids].sum()),age_median=median,strata=json.dumps(allocations)))
    return ids
def make_model(family,params,strategy,b,seed):
    weighted=strategy=="class_weights";ratio=float(sum(b==0)/sum(b==1))
    if family=="LR":return LogisticRegression(**params,l1_ratio=0,solver="lbfgs",max_iter=3000,class_weight="balanced" if weighted else None,random_state=seed)
    if family=="RF":return RandomForestClassifier(**params,class_weight="balanced" if weighted else None,random_state=seed,n_jobs=2)
    if family=="CatBoost":return CatBoostClassifier(**params,loss_function="Logloss",verbose=False,allow_writing_files=False,thread_count=2,random_seed=seed,auto_class_weights="Balanced" if weighted else None)
    return XGBClassifier(**params,objective="binary:logistic",eval_metric="logloss",tree_method="hist",n_jobs=2,random_state=seed,scale_pos_weight=ratio if weighted else 1.0)
def train(a,b,family,cfg,seed):
    pre=make_preprocessor(list(a.columns),family,cfg["indicators"])
    z=pre.fit_transform(a)
    assert np.isfinite(z).all()
    names=pre.get_feature_names_out().tolist()
    assert "numeric__crp" in names, names
    assert "missing__missingindicator_crp" not in names, names
    assert not any(name.startswith("missing__") and name.endswith("_crp") for name in names), names
    expected=["missing__missingindicator_"+c for c in a.columns if c != "crp"] if cfg["indicators"] else []
    assert [name for name in names if name.startswith("missing__")]==expected
    ids=sample_indices(a,b,cfg["strategy"],seed)
    model=make_model(family,cfg["params"],cfg["strategy"],b,seed)
    model.fit(z[ids],b[ids]);assert list(model.classes_)==[0,1]
    return {"preprocessor":pre,"model":model,"features":list(a.columns),"family":family,"config":cfg,"run_id":RUN_ID}
def predict(artifact,a):
    p=artifact["model"].predict_proba(artifact["preprocessor"].transform(a))[:,1]
    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    return p
def threshold(b,p):
    fpr,tpr,t=roc_curve(b,p,drop_intermediate=False)
    score=(tpr+1-fpr)/2;valid=np.isfinite(t)
    mx=score[valid].max();ids=np.flatnonzero(valid&np.isclose(score,mx,atol=1e-12,rtol=0))
    idx=ids[np.argmax(tpr[ids])];return float(t[idx])
def metrics(b,p,t):
    h=p>=t;tn,fp,fn,tp=confusion_matrix(b,h,labels=[0,1]).ravel()
    return dict(AP=average_precision_score(b,p),ROC_AUC=roc_auc_score(b,p),Brier=brier_score_loss(b,p),sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),precision=precision_score(b,h,zero_division=0),F1=f1_score(b,h,zero_division=0),accuracy=accuracy_score(b,h),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),n=len(b),deaths=int(sum(b)),mean_probability=float(np.mean(p)))
def suggest(trial,family):
    if family=="LR":return {"C":trial.suggest_float("C",1e-3,100,log=True)}
    if family=="RF":return {"n_estimators":trial.suggest_int("n_estimators",200,800,step=100),"max_depth":trial.suggest_int("max_depth",2,8),"min_samples_leaf":trial.suggest_categorical("min_samples_leaf",[1,2,4,8,16]),"max_features":trial.suggest_categorical("max_features",["sqrt",0.5,1.0])}
    if family=="CatBoost":return {"iterations":trial.suggest_int("iterations",100,800,step=100),"depth":trial.suggest_int("depth",2,6),"learning_rate":trial.suggest_float("learning_rate",.01,.2,log=True),"l2_leaf_reg":trial.suggest_float("l2_leaf_reg",.01,100,log=True)}
    return {"n_estimators":trial.suggest_int("n_estimators",100,800,step=100),"max_depth":trial.suggest_int("max_depth",1,5),"learning_rate":trial.suggest_float("learning_rate",.01,.2,log=True),"min_child_weight":trial.suggest_float("min_child_weight",1,20),"subsample":trial.suggest_float("subsample",.6,1),"colsample_bytree":trial.suggest_float("colsample_bytree",.6,1),"reg_lambda":trial.suggest_float("reg_lambda",.001,100,log=True),"reg_alpha":trial.suggest_float("reg_alpha",0,5)}
def inner_predictions(a,b,g,family,cfg,ss,seed):
    p=np.full(len(b),np.nan);aps=[];bs=[]
    for k,(tr,va) in enumerate(ss):
        artifact=train(a.iloc[tr],b[tr],family,cfg,seed+k)
        p[va]=predict(artifact,a.iloc[va]);aps.append(average_precision_score(b[va],p[va]));bs.append(brier_score_loss(b[va],p[va]))
    assert np.isfinite(p).all()
    return p,float(np.mean(aps)),float(np.mean(bs))
def tune(a,b,g,family,ss,name,seed,trials=None,fixed=None):
    if trials is None: trials=CONFIG["main_trials"]
    study_name=RUN_ID+"_"+name+"_"+family
    sampler=optuna.samplers.TPESampler(seed=seed,multivariate=False,constant_liar=False)
    study=optuna.create_study(study_name=study_name,storage=STORAGE,direction="maximize",sampler=sampler,load_if_exists=True)
    if not study.trials:
        options=[(s,i) for s in STRATEGIES for i in [False,True]] if fixed is None else [(fixed["strategy"],fixed["indicators"])]
        for s,i in options:study.enqueue_trial({**BASE[family],"strategy":s,"indicators":i})
        study.set_user_attr("config",CONFIG);study.set_user_attr("features",list(a.columns))
    def objective(trial):
        try:
            cfg={"strategy":trial.suggest_categorical("strategy",STRATEGIES) if fixed is None else fixed["strategy"],"indicators":trial.suggest_categorical("indicators",[False,True]) if fixed is None else fixed["indicators"],"params":suggest(trial,family)}
            p,ap,br=inner_predictions(a,b,g,family,cfg,ss,seed)
            trial.set_user_attr("config",cfg);trial.set_user_attr("Brier",br)
            return ap
        except Exception as exc:
            trial.set_user_attr("error",repr(exc));raise
    done=len([t for t in study.trials if t.state!=optuna.trial.TrialState.WAITING])
    if done<trials:study.optimize(objective,n_trials=trials-done,n_jobs=1,catch=(Exception,))
    complete=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    assert complete,study_name
    maximum=max(t.value for t in complete);eligible=[t for t in complete if maximum-t.value<=1e-6]
    best=min(eligible,key=lambda t:(t.user_attrs["Brier"],t.number))
    cfg=best.user_attrs["config"]
    p,ap,br=inner_predictions(a,b,g,family,cfg,ss,seed)
    log(f"{name} {family}: {len(complete)}/{trials} complete; inner AP={ap:.5f}; {cfg}")
    return {"family":family,"config":cfg,"inner_AP":ap,"inner_Brier":br,"threshold":threshold(b,p),"study_name":study_name,"best_trial":best.number,"completed":len(complete)}
def choose(results):
    mx=max(r["inner_AP"] for r in results)
    return min([r for r in results if mx-r["inner_AP"]<=1e-6],key=lambda r:(r["inner_Brier"],FAMILIES.index(r["family"])))
def records(ids,r,f,family,p,t,cfg):
    return [dict(row_id=int(D.row_id.iloc[j]),group_id=int(groups[j]),repeat=r,fold=f,family=family,y_true=int(y[j]),p_death=float(prob),threshold=t,run_id=RUN_ID,configuration=json.dumps(cfg,sort_keys=True)) for j,prob in zip(ids,p)]


