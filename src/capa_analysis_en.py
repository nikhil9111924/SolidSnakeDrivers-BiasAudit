import json, os, math
import pandas as pd, numpy as np

BASELINE_PATH = "results/baseline_with_completions.csv"
COMPARES = {
    "self_debias": "results/mitigated_results.csv"
}

def softmax_pair(a, b):
    m = max(a, b)
    ea, eb = math.exp(a - m), math.exp(b - m)
    s = ea + eb
    return ea / s, eb / s

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
    for a, b in zip(df["log_prob_stereotypical"].tolist(), df["log_prob_anti"].tolist() if "log_prob_anti" in df.columns else df["log_prob_anti_stereotypical"].tolist()):
        p_s, p_a = softmax_pair(a, b)
        ps.append(p_s); pa.append(p_a)
    df = df.copy()
    df["p_stereotypical"] = ps
    df["p_anti"] = pa
    df["key"] = df["template_id"].astype(str) + "|" + df["demographics"]
    return df

def run_one(baseline, comp_df, name, out_dir="results"):
    merged = baseline.merge(comp_df, on="key", suffixes=("_base", "_comp"))
    if len(merged) == 0:
        print(f"[WARN] No overlap for {name}.")
        return None

    p1 = merged[["p_stereotypical_base","p_anti_base"]].to_numpy()
    p2 = merged[["p_stereotypical_comp","p_anti_comp"]].to_numpy()

    correct_idx = np.ones(len(merged), dtype=int)  # anti index=1
    kappa, A_obs, A_exp = capa(p1, p2, correct_idx)

    out = {"N": int(len(merged)), "kappa_p": float(kappa), "A_obs": float(A_obs), "A_exp": float(A_exp)}
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"capa_summary_en_{name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote {out_path}")
    return out

def main():
    if not os.path.exists(BASELINE_PATH):
        print(f"[ERR] Missing {BASELINE_PATH}")
        return
    base = pd.read_csv(BASELINE_PATH)
    base = attach_probs(base)

    for name, path in COMPARES.items():
        if not os.path.exists(path):
            print(f"[WARN] Missing {path} — skipping {name}")
            continue
        comp = pd.read_csv(path)
        comp = attach_probs(comp)
        run_one(base, comp, name)

if __name__ == "__main__":
    main()
