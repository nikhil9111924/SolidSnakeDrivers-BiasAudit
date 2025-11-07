#!/usr/bin/env python3
import json, math, os
import pandas as pd, numpy as np

BASELINE_PATH = "results/baseline_with_completions_hi.csv"
MITIGATIONS = {
    "self-debias": "results/mitigation_results_hi_self-debias.csv",
    "safety":      "results/mitigation_results_hi_safety.csv",
}

def softmax_pair(a,b):
    m=max(a,b); ea,eb=math.exp(a-m), math.exp(b-m); s=ea+eb
    return ea/s, eb/s

def _norm_key(tid, demo_json_str):
    d = json.loads(demo_json_str)
    return f"{tid}|{json.dumps(d, sort_keys=True, ensure_ascii=False)}"

def attach_probs(df):
    anti_col = "log_prob_anti" if "log_prob_anti" in df.columns else "log_prob_anti_stereotypical"
    ps, pa = [], []
    for a, b in zip(df["log_prob_stereotypical"].tolist(), df[anti_col].tolist()):
        p_s, p_a = softmax_pair(a,b); ps.append(p_s); pa.append(p_a)
    out = df.copy()
    out["p_stereotypical"] = ps; out["p_anti"] = pa
    out["key"] = [_norm_key(t, d) for t, d in zip(out["template_id"], out["demographics"])]
    return out

def capa(p1, p2, correct_idx=None):
    p1 = p1 / p1.sum(axis=1, keepdims=True); p2 = p2 / p2.sum(axis=1, keepdims=True)
    A_obs = float(np.mean(np.sum(p1 * p2, axis=1)))
    N, K = p1.shape
    if correct_idx is None:
        alpha1 = float(np.mean(np.max(p1, axis=1))); alpha2 = float(np.mean(np.max(p2, axis=1)))
        correct_idx = np.argmax(p1, axis=1)
    else:
        alpha1 = float(np.mean(p1[np.arange(N), correct_idx])); alpha2 = float(np.mean(p2[np.arange(N), correct_idx]))
    A_exp_terms=[]
    for i in range(N):
        c=int(correct_idx[i])
        q1=np.full(K,(1.0-alpha1)/(K-1)); q2=np.full(K,(1.0-alpha2)/(K-1))
        q1[c]=alpha1; q2[c]=alpha2
        A_exp_terms.append(float(np.dot(q1,q2)))
    A_exp=float(np.mean(A_exp_terms))
    return (A_obs - A_exp) / (1.0 - A_exp + 1e-12), A_obs, A_exp

def run_one(baseline, mitig_df, name, out_dir="results"):
    merged = baseline.merge(mitig_df, on="key", suffixes=("_base", "_mitig"))
    if len(merged)==0:
        print(f"[WARN] No overlap for {name}."); return None
    p1 = merged[["p_stereotypical_base","p_anti_base"]].to_numpy()
    p2 = merged[["p_stereotypical_mitig","p_anti_mitig"]].to_numpy()
    kappa, A_obs, A_exp = capa(p1, p2, np.ones(len(merged), dtype=int))  # anti index=1

    summaries = {"overall":{"N":int(len(merged)), "kappa_p":float(kappa), "A_obs":float(A_obs), "A_exp":float(A_exp)}}
    axes = ["gender","profession","nationality","age"]
    demo_objs = merged["demographics_base"].apply(json.loads)
    for ax in axes:
        vals = sorted({d.get(ax) for d in demo_objs if ax in d})
        for v in vals:
            mask = demo_objs.apply(lambda d: d.get(ax)==v)
            if mask.sum()==0: continue
            sub_p1 = p1[mask.values]; sub_p2 = p2[mask.values]
            sk, so, se = capa(sub_p1, sub_p2, np.ones(len(sub_p1), dtype=int))
            summaries[f"{ax}={v}"] = {"N":int(mask.sum()), "kappa_p":float(sk), "A_obs":float(so), "A_exp":float(se)}

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"capa_summary_hi_{name}.json")
    with open(out_path,"w",encoding="utf-8") as f: json.dump(summaries,f,indent=2,ensure_ascii=False)
    print(f"[OK] Wrote {out_path}")
    return out_path, summaries

def main():
    if not os.path.exists(BASELINE_PATH):
        print(f"[ERR] Missing {BASELINE_PATH}"); return
    baseline = attach_probs(pd.read_csv(BASELINE_PATH))

    results={}
    for name, path in MITIGATIONS.items():
        if not os.path.exists(path):
            print(f"[WARN] Missing {path} — skipping {name}"); continue
        mitig = attach_probs(pd.read_csv(path))
        out = run_one(baseline, mitig, name)
        results[name] = out[1] if out else None

    with open("results/capa_summary_hi_ALL.json","w",encoding="utf-8") as f:
        json.dump(results,f,indent=2,ensure_ascii=False)
    print("[DONE] -> results/capa_summary_hi_ALL.json")

if __name__ == "__main__":
    main()
