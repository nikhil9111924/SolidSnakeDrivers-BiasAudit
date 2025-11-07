#!/usr/bin/env python3
"""
analyze_bias_percentage.py
Compute the share of rows with stereotype preference (score > 0).

Now supports:
  * CSV output via --out
  * JSON to stdout via --json-stdout (for Colab piping)
Usage:
  python src/analyze_bias_percentage.py --input <csv> --axis gender --out results/foo.csv
  python src/analyze_bias_percentage.py --input <csv> --axis gender --json-stdout
"""
import argparse, json, os
import pandas as pd

def compute(df: pd.DataFrame, axis: str | None):
    df = df.copy()
    df["is_stereo"] = df["stereotype_preference_score"] > 0
    rows = [{"slice": "overall", "n": int(len(df)),
             "pct_stereotype": float(df["is_stereo"].mean())}]
    if axis:
        j = df["demographics"].astype(str).apply(json.loads)
        for v, sub in df.groupby(j.apply(lambda d: d.get(axis, "UNKNOWN"))):
            rows.append({"slice": f"{axis}={v}", "n": int(len(sub)),
                         "pct_stereotype": float(sub["is_stereo"].mean())})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--axis", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json-stdout", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    rows = compute(df, args.axis)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"[OK] wrote {args.out}")

    if args.json_stdout or not args.out:
        # Print a single JSON object (list of rows)
        print(json.dumps(rows, ensure_ascii=False))

if __name__ == "__main__":
    main()
