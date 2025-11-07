import json
import math
import os
import pandas as pd
import numpy as np

BASELINE_PATH = "results/baseline_with_completions_hi.csv"
MITIGATIONS = {
    "self_debias": "results/mitigation_results_hi_self-debias.csv",
    "safety": "results/mitigation_results_hi_safety.csv"
}

def softmax_pair(a, b):
    m = max(a, b)
    ea, eb = math.exp(a - m), math.exp(b - m)
    s = ea + eb
    return ea / s, eb / s

def parse_demo(s):
    try:
        return json.loads(s)
    except Exception:
        return {}

def capa(p1: np.ndarray, p2: np.ndarray, correct_idx=None):
    p1 = p1 / p1.sum(axis=1, keepdims=True)
    p2 = p2 / p2.sum(axis=1, keepdims=True)
    A_obs = float(np.mean(np.sum(p1 * p2, axis=1)))

    N, K = p1.shape
    if correct_idx is None:
        alpha1 = float(np.mean(np.max(p1, axis=1)))
        alpha2 = float(np.mean(np.max(p2, axis=1)))
        correct_idx = np.argmax(p1, axis=1)
    else:
        alpha1 = float(np.mean(p1[np.arange(N), correct_idx]))
        alpha2 = float(np.mean(p2[np.arange(N), correct_idx]))

    A_exp_terms = []
    for i in range(N):
        c = int(correct_idx[i])
        q1 = np.full(K, (1.0 - alpha1) / (K - 1))
        q2 = np.full(K, (1.0 - alpha2) / (K - 1))
        q1[c] = alpha1
        q2[c] = alpha2
        A_exp_terms.append(float(np.dot(q1, q2)))
    A_exp = float(np.mean(A_exp_terms))
    denom = (1.0 - A_exp) if (1.0 - A_exp) != 0 else 1e-12
    kappa_p = (A_obs - A_exp) / denom
    return kappa_p, A_obs, A_exp

def attach_probs(df):
    ps, pa = [], []
    for a, b in zip(df["log_prob_stereotypical"].tolist(), df["log_prob_anti"].tolist()):
        p_s, p_a = softmax_pair(a, b)
        ps.append(p_s); pa.append(p_a)
    df = df.copy()
    df["p_stereotypical"] = ps
    df["p_anti"] = pa
    df["key"] = df["template_id"].astype(str) + "|" + df["demographics"]
    return df

def run_one(baseline, mitig_df, name, out_dir="results"):
    merged = baseline.merge(mitig_df, on="key", suffixes=("_base", "_mitig"))
    if len(merged) == 0:
        print(f"[WARN] No overlap for {name}.")
        return None

    p1 = merged[["p_stereotypical_base","p_anti_base"]].to_numpy()
    p2 = merged[["p_stereotypical_mitig","p_anti_mitig"]].to_numpy()

    # Correct = anti-stereotypical -> index 1
    correct_idx = np.ones(len(merged), dtype=int)
    kappa, A_obs, A_exp = capa(p1, p2, correct_idx)

    # per-axis breakdown (gender, profession, nationality, age if present)
    summaries = {"overall": {"N": int(len(merged)), "kappa_p": float(kappa), "A_obs": float(A_obs), "A_exp": float(A_exp)}}
    axes = ["gender","profession","nationality","age"]
    demo_objs = merged["demographics_base"].apply(parse_demo)
    for ax in axes:
        vals = sorted({d.get(ax) for d in demo_objs if ax in d})
        for v in vals:
            mask = demo_objs.apply(lambda d: d.get(ax)==v)
            if mask.sum() == 0:
                continue
            sub_p1 = p1[mask.values]
            sub_p2 = p2[mask.values]
            sub_k, sub_Ao, sub_Ae = capa(sub_p1, sub_p2, np.ones(len(sub_p1), dtype=int))
            summaries[f"{ax}={v}"] = {"N": int(mask.sum()), "kappa_p": float(sub_k), "A_obs": float(sub_Ao), "A_exp": float(sub_Ae)}

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"capa_summary_hi_{name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote {out_path}")
    return out_path, summaries

def main():
    if not os.path.exists(BASELINE_PATH):
        print(f"[ERR] Missing {BASELINE_PATH}")
        return
    baseline = pd.read_csv(BASELINE_PATH)
    baseline = attach_probs(baseline)

    results = {}
    for name, path in MITIGATIONS.items():
        if not os.path.exists(path):
            print(f"[WARN] Missing {path} — skipping {name}")
            continue
        mitig = pd.read_csv(path)
        mitig = attach_probs(mitig)
        out = run_one(baseline, mitig, name)
        results[name] = out[1] if out else None

    with open("results/capa_summary_hi_ALL.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("[DONE] summaries -> results/capa_summary_hi_ALL.json")

if __name__ == "__main__":
    main()
