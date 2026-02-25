import numpy as np
import itertools
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
hair_cols = [c for c in df.columns if "hair" in c.lower()]
marker_cols = [c for c in df.columns if "marker" in c.lower()]

def agreement_stats(df, cols):
    X = df[cols]
    
    full_agree = (X.nunique(axis=1) == 1)
    full_agree_pct = 100 * full_agree.mean()

    majority = X.apply(lambda row: row.value_counts().max() >= 4, axis=1)
    majority_pct = 100 * majority.mean()

    pairwise = {}
    for c1, c2 in itertools.combinations(cols, 2):
        pairwise[(c1, c2)] = 100 * (X[c1] == X[c2]).mean()

    avg_pairwise = np.mean(list(pairwise.values()))

    return full_agree_pct, majority_pct, avg_pairwise

hair_full, hair_majority, hair_avgpair = agreement_stats(df, hair_cols)
marker_full, marker_majority, marker_avgpair = agreement_stats(df, marker_cols)

print("HAIR (0–3)")
print(f"Full agreement (all 5 same): {hair_full:.2f}%")
print(f"Majority agreement (>=4 agree): {hair_majority:.2f}%")
print(f"Average pairwise agreement: {hair_avgpair:.2f}%")

print("MARKER (0–1)")
print(f"Full agreement (all 5 same): {marker_full:.2f}%")
print(f"Majority agreement (>=4 agree): {marker_majority:.2f}%")
print(f"Average pairwise agreement: {marker_avgpair:.2f}%")