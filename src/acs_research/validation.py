"""Independent checks of saved ACS predictions, splits, selection and SHAP.

This audits numerical artifacts without rerunning the expensive Optuna searches.
It does not establish clinical transportability or anonymity of patient-level data.
"""
from pathlib import Path
import hashlib
import json
import sqlite3
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix


def independent_metrics(frame):
    y = frame.y_true.to_numpy()
    p = frame.p_death.to_numpy()
    decision = p >= frame.threshold.to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, decision, labels=[0, 1]).ravel()
    return dict(AP=float(average_precision_score(y, p)),
                ROC_AUC=float(roc_auc_score(y, p)), Brier=float(np.mean((p-y)**2)),
                sensitivity=float(tp/(tp+fn)), specificity=float(tn/(tn+fp)),
                precision=float(tp/(tp+fp)) if tp+fp else 0.,
                F1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,
                accuracy=float((tp+tn)/len(y)), TN=int(tn), FP=int(fp), FN=int(fn),
                TP=int(tp), n=len(y), deaths=int(y.sum()), mean_probability=float(p.mean()))


def validate_run(directory: Path, *, check_model: bool = True) -> dict:
    directory = Path(directory)
    checks = 0
    def checked(condition, label):
        nonlocal checks
        if not bool(condition):
            raise ValueError("Validation failed: " + label)
        checks += 1

    protocol = json.loads((directory / "02_протокол.json").read_text())
    rid = protocol["run_id"]
    data = pd.read_csv(directory / "01_данные.csv").set_index("row_id")
    checked(len(data) == 906 and data.hospital_death.sum() == 43 and
            data.group_id.nunique() == 892, "cohort, deaths and groups")
    checked(data.crp.isna().sum() == 48, "CRP missing observations retained")
    checked(hashlib.sha256((directory / "01_данные.csv").read_bytes()).hexdigest()
            == protocol["data_sha256"], "prepared data fingerprint")
    outer = pd.read_csv(directory / "02_внешние_разбиения.csv")
    inner = pd.read_csv(directory / "02_внутренние_разбиения.csv")
    checked(len(outer.groupby(["repeat", "fold"])) == 9, "nine outer splits")
    for (repeat, fold), block in outer.groupby(["repeat", "fold"]):
        train = block[block.side == "train"]
        test = block[block.side == "test"]
        checked(set(train.group_id).isdisjoint(test.group_id), "outer group separation")
        checked(set(train.row_id).isdisjoint(test.row_id) and
                set(train.row_id) | set(test.row_id) == set(data.index), "outer partition")
        local = inner[(inner.repeat == repeat) & (inner.fold == fold)]
        checked(local.inner_fold.nunique() == 3, "three inner splits")
        for _, split in local.groupby("inner_fold"):
            it = split[split.side == "train"]
            iv = split[split.side == "validation"]
            checked(set(it.group_id).isdisjoint(iv.group_id), "inner group separation")
            checked(set(it.row_id).isdisjoint(iv.row_id) and
                    set(it.row_id) | set(iv.row_id) == set(train.row_id), "inner partition")

    max_error = 0.
    frames = {}
    for tag, oof_name, fold_name, repeat_name, key in [
        ("main", "02_oof_predictions.csv", "02_метрики_блоков.csv", "02_метрики_повторений.csv", "family"),
        ("conditional", "02_условные_oof.csv", "02_условные_метрики_блоков.csv", "02_условные_метрики_повторений.csv", "branch")]:
        frame = pd.read_csv(directory / oof_name)
        checked(set(frame.run_id) == {rid}, tag + " matching run")
        checked(frame.p_death.between(0, 1).all() and np.isfinite(frame.p_death).all(), tag + " probabilities")
        checked(not frame.duplicated(["repeat", "family", "row_id"]).any(), tag + " unique OOF")
        checked(frame.groupby(["repeat", "family"]).size().eq(len(data)).all(), tag + " complete OOF")
        checked(np.array_equal(frame.y_true, data.loc[frame.row_id, "hospital_death"]), tag + " labels")
        checked(np.array_equal(frame.group_id, data.loc[frame.row_id, "group_id"]), tag + " groups")
        for (repeat, fold, family), block in frame.groupby(["repeat", "fold", "family"]):
            expected = outer[(outer.repeat == repeat) & (outer.fold == fold) & (outer.side == "test")]
            checked(set(block.row_id) == set(expected.row_id), tag + " unchanged test patients")
        for filename, pooled in [(fold_name, False), (repeat_name, True)]:
            saved = pd.read_csv(directory / filename)
            for _, row in saved.iterrows():
                family = row["name"] if pooled else row[key]
                block = frame[(frame.repeat == row["repeat"]) & (frame.family == family)]
                if not pooled:
                    block = block[block.fold == row["fold"]]
                for metric, value in independent_metrics(block).items():
                    tolerance = 1e-7 if metric in {"Brier", "mean_probability"} else 1e-12
                    error = abs(value-float(row[metric]))
                    max_error = max(max_error, error)
                    checked(error <= tolerance, tag + " independently recalculated " + metric)
        frames[tag] = frame
        comparison_file = "02_сравнение_моделей.csv" if tag == "main" else "02_условные_сравнения.csv"
        comparison = pd.read_csv(directory / comparison_file).set_index("name")
        repeats = pd.read_csv(directory / repeat_name)
        for metric in ["AP", "ROC_AUC", "Brier"]:
            calculated = repeats.groupby("name")[metric].mean().loc[comparison.index]
            checked(np.allclose(calculated, comparison[metric], atol=1e-12, rtol=0), tag + " published summary " + metric)

    search = json.loads((directory / "02_результаты_поиска.json").read_text())
    for (repeat, fold), block in frames["main"][frames["main"].family == "selector"].groupby(["repeat", "fold"]):
        candidates = [r for r in search if r["repeat"] == repeat and r["fold"] == fold]
        best_ap = max(r["inner_AP"] for r in candidates)
        winner = min([r for r in candidates if best_ap-r["inner_AP"] <= 1e-6],
                     key=lambda r: (r["inner_Brier"], protocol["families"].index(r["family"])))
        corresponding = frames["main"][(frames["main"].repeat == repeat) &
                                       (frames["main"].fold == fold) &
                                       (frames["main"].family == winner["family"])].set_index("row_id").loc[block.row_id]
        checked(np.array_equal(corresponding.p_death, block.p_death), "selector forecasts")
        checked(np.array_equal(corresponding.threshold, block.threshold), "selector thresholds")
    final = json.loads((directory / "02_final_config.json").read_text())
    candidates = final["all_families"]
    best_ap = max(r["inner_AP"] for r in candidates)
    best = min([r for r in candidates if best_ap-r["inner_AP"] <= 1e-6],
               key=lambda r: (r["inner_Brier"], protocol["families"].index(r["family"])))
    checked(best == final["selection"] and final["run_id"] == rid, "final inner selection")
    database = directory / "02_optuna_без_индикатора_СРБ.sqlite"
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as conn:
        studies = conn.execute("SELECT study_id, study_name FROM studies").fetchall()
        matching = [(sid, name) for sid, name in studies if name.startswith(rid + "_")]
        checked(len(matching) == 112, "112 matching studies")
        completed = 0
        for sid, name in matching:
            counts = dict(conn.execute("SELECT state, COUNT(*) FROM trials WHERE study_id=? GROUP BY state", (sid,)).fetchall())
            expected = protocol["conditional_trials"] if "_conditional_" in name else protocol["main_trials"]
            checked(counts == {"COMPLETE": expected}, "trial budget and completion")
            completed += expected

    sampling = pd.read_csv(directory / "02_аудит_балансировки.csv")
    checked(sampling.selected_deaths.eq(sampling.train_deaths).all(), "all training deaths retained")
    sampling = sampling[sampling.strategy.str.contains("4to1")]
    checked(sampling.selected_n.eq(5*sampling.selected_deaths).all(), "4:1 sampling")
    additivity = None
    if check_model:
        import joblib
        model_file = directory / "02_final_model.joblib"
        artifact = joblib.load(model_file)
        checked(artifact["run_id"] == rid, "model run")
        names = artifact["preprocessor"].get_feature_names_out().tolist()
        checked("numeric__crp" in names and "missing__missingindicator_crp" not in names, "CRP value without CRP missing flag")
        transformed = artifact["preprocessor"].transform(data[artifact["features"]])
        model = artifact["model"]
        p = model.predict_proba(transformed)[:, 1]
        with np.load(directory / "03_shap_values.npz", allow_pickle=False) as values:
            checked(values["run_id"].tolist() == [rid], "SHAP run")
            checked(values["model_sha256"].tolist() == [hashlib.sha256(model_file.read_bytes()).hexdigest()], "SHAP model hash")
            checked(np.max(abs(p-values["p_death"])) < 1e-12, "SHAP predictions")
            family = artifact["family"]
            if family == "LR": output = model.decision_function(transformed)
            elif family == "RF": output = p
            elif family == "CatBoost": output = model.predict(transformed, prediction_type="RawFormulaVal")
            else: output = model.predict(transformed, output_margin=True)
            checked(np.max(abs(output-values["model_output"])) < 1e-12, "SHAP output scale")
            additivity = float(np.max(abs(values["base_values"]+values["values"].sum(axis=1)-output)))
            checked(additivity < 1e-5, "SHAP additivity")
            checked(values["transformed_feature_names"].tolist() == names, "SHAP feature order")
    return dict(status="passed", run_id=rid, checks=checks, studies=len(matching),
                completed_trials=completed, max_metric_difference=max_error,
                max_shap_additivity_error=additivity)
