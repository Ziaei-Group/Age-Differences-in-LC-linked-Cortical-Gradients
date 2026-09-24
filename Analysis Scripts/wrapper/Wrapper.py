"""Wrapper: computes the main results of the paper from the released data
=======================================================================

Runs the statistics reported in the paper on the master sheet
(Source Data/Disperion_source_data/All_in_1_MasterSheet.xlsx; 140 participants:
72 younger, 68 older adults), prints them, and writes one CSV per analysis to
the folder  replication_output/  at the repository root.

It calls code that is already in this repository:
  - age_dispersion_behaviour_functions.py  (age comparisons, brain-behaviour
    correlations with permutation tests, age-moderation models)
  - PT seed age difference.ipynb           (pontine-tegmentum control seed)
  - valence arousal behaviour.ipynb        (valence and arousal ratings)
  - Source Data/Sensitivity_Analyses_source_data  (saved split-half, 5-fold and
    leave-one-out reliability results; summarised, not recomputed)


Covariates, as in the paper:
  - age comparisons: sex, head motion (mean FD of the same condition) and mean
    cortical thickness; the seven networks are Bonferroni-corrected (x7)
  - brain-behaviour: sex and head motion; 10,000 permutations, seed 42
Sex enters as the numeric 'Sex' column (1 = female); the text 'Gender' column
(F/M) is only used to check it.

How to run
  1. Download or clone the repository.
  2. pip install pandas numpy scipy statsmodels openpyxl matplotlib seaborn
  3. python "Analysis Scripts/wrapper/Wrapper.py"      (from any folder; ~10 s)
Input files are never modified. Re-running overwrites replication_output/.
"""

import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.io

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None          # the author functions call plt.show()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                                  # repository root
SCRIPTS = ROOT / "Analysis Scripts"
MASTER = ROOT / "Source Data" / "Disperion_source_data" / "All_in_1_MasterSheet.xlsx"
SENS = ROOT / "Source Data" / "Sensitivity_Analyses_source_data"
OUTPUT = ROOT / "replication_output"

sys.path.insert(0, str(SCRIPTS))
import age_dispersion_behaviour_functions as author       # noqa: E402

N_PERM = 10_000
SEED = 42
COV_AGE = ["Sex", "Mean_FD_Negative", "Cortical_Thickness"]
COV_AGE_NEUTRAL = ["Sex", "Mean_FD_Neutral", "Cortical_Thickness"]
COV_BEH = ["Sex", "Mean_FD_Negative"]
LOO = ["Right_G1_Negative_LOO", "Right_G2_Negative_LOO"]
NETWORKS = {1: "VIS", 2: "SMN", 3: "DAN", 4: "VAN", 5: "LIM", 6: "FPN", 7: "DMN"}


def load_master():
    m = pd.read_excel(MASTER)
    assert len(m) == 140 and m.Sub.is_unique, "expected 140 unique participants"
    assert m.Age_Cat.value_counts().to_dict() == {"Young": 72, "Old": 68}
    assert m.Gender.eq(m.Sex.map({0: "M", 1: "F"})).all(), "Sex (1=F) and Gender disagree"
    return m


def age_test(data, feature, cov, label, bonf=None):
    r = author.group_comparison(data, feature, cov)
    out = dict(label=label, feature=feature, t=r["t_statistic"], p=r["p_value"],
               d=abs(r["cohens_d"]), ci_lower=r["ci_lower"], ci_upper=r["ci_upper"],
               n_young=r["n_group1"], n_old=r["n_group2"], covariates="+".join(cov))
    if bonf:
        out["p_bonf"] = min(1.0, r["p_value"] * bonf)
    print(f"  {label:24s} t = {out['t']:6.2f}  p = {out['p']:.3f}  d = {out['d']:.2f}"
          + (f"  p_bonf = {out['p_bonf']:.3f}" if bonf else ""))
    return out


def behaviour(data, feature, outcome, cov, label):
    r = author.plot_correlation_by_group(data, feature, outcome, cov, feature, outcome,
                                         n_perm=N_PERM, rng=np.random.default_rng(SEED))
    plt.close("all")
    m = r["moderation"]
    print(f"  {label:24s} age x dispersion beta = {m['beta_std']:.3f}  p = {m['p_value']:.3f}")
    out = [dict(label=label, feature=feature, outcome=outcome, group="interaction", n=np.nan,
                r=np.nan, p_perm=np.nan, ci_lower=m["ci_lower"], ci_upper=m["ci_upper"],
                beta_std=m["beta_std"], p_value=m["p_value"])]
    for g in ("Young", "Old"):
        s = r[g]
        print(f"  {'':24s} {g:5s} r = {s['r']:6.3f}  p_perm = {s['p_perm']:.3f}  "
              f"95% CI [{s['ci_lower']:.3f}, {s['ci_upper']:.3f}]  n = {s['n']}")
        out.append(dict(label=label, feature=feature, outcome=outcome, group=g, n=s["n"],
                        r=s["r"], p_perm=s["p_perm"], ci_lower=s["ci_lower"],
                        ci_upper=s["ci_upper"], beta_std=np.nan, p_value=s["p_analytic"]))
    return out


def notebook_source(name):
    nb = json.loads((SCRIPTS / name).read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def run(data):
    tables = {}

    print("\nDemographics")
    dem = data.groupby("Age_Cat").agg(n=("Age", "size"), mean=("Age", "mean"), sd=("Age", "std"),
                                      min=("Age", "min"), max=("Age", "max"), females=("Sex", "sum"))
    print(dem.round(2).to_string())
    tables["demographics"] = dem.reset_index()

    print("\nAge differences, negative movie (sex + FD + thickness)")
    res = []
    for side, suf in (("R", ""), ("L", "_Left")):
        res.append(age_test(data, "Global_Dispersion" + suf, COV_AGE, f"{side} Global"))
        res.append(age_test(data, "Between_network_Dispersion" + suf, COV_AGE, f"{side} Between"))
        for k, net in NETWORKS.items():
            res.append(age_test(data, f"Dispersion_N{k}{suf}", COV_AGE, f"{side} {net}", bonf=7))
    print("\nGradient range")
    for side, s in (("R", "Right"), ("L", "Left")):
        for g in ("G1", "G2"):
            res.append(age_test(data, f"Study_specific_{s}_LC_Range_{g}", COV_AGE, f"{side} Range {g}"))
    print("\nAge differences, neutral movie")
    for side, s in (("R", "Right"), ("L", "Left")):
        res.append(age_test(data, f"Global_Dispersion_{s}_Neutral", COV_AGE_NEUTRAL, f"{side} Global neutral"))
        res.append(age_test(data, f"Between_network_Dispersion_{s}_Neutral", COV_AGE_NEUTRAL, f"{side} Between neutral"))
        for k, net in NETWORKS.items():
            res.append(age_test(data, f"Dispersion_N{k}_{s}_Neutral", COV_AGE_NEUTRAL, f"{side} {net} neutral"))
    tables["age_comparisons"] = pd.DataFrame(res)

    print("\nAge differences with LOO reliability as extra covariates")
    tables["age_comparisons_loo"] = pd.DataFrame([
        age_test(data, "Global_Dispersion", COV_AGE + LOO, "R Global +LOO"),
        age_test(data, "Between_network_Dispersion", COV_AGE + LOO, "R Between +LOO"),
        age_test(data, "Dispersion_N6", COV_AGE + LOO, "R FPN +LOO")])

    print(f"\nBrain-behaviour (sex + FD; {N_PERM:,} permutations, seed {SEED})")
    beh = []
    for feat, lab in (("Global_Dispersion", "Global"), ("Dispersion_N6", "FPN"),
                      ("Dispersion_N1", "VIS"), ("Dispersion_N5", "LIM")):
        beh += behaviour(data, feat, "Emotional_Resilience_Index", COV_BEH, f"{lab} - ERI")
    beh += behaviour(data, "Dispersion_N1", "Cognitive_PC", COV_BEH, "VIS - Cognition")
    tables["brain_behaviour"] = pd.DataFrame(beh)

    print("\nModeration with LOO reliability as extra covariates")
    mod = []
    for feat, lab in (("Global_Dispersion", "Global"), ("Dispersion_N6", "FPN")):
        m = author.test_age_moderation(data, feat, "Emotional_Resilience_Index", COV_BEH + LOO)
        print(f"  {lab:24s} beta = {m['beta_std']:.3f}  p = {m['p_value']:.3f}")
        mod.append(dict(label=lab, feature=feat, **m))
    tables["moderation_loo"] = pd.DataFrame(mod)

    print("\nPontine-tegmentum control seed")
    pairs = {"Global_Dispersion": "global_dispersion_PT",
             "Between_network_Dispersion": "between__dispersion_PT",
             "Dispersion_N6": "FPN_dispersion_PT"}
    names = {"Global_Dispersion": "Global", "Between_network_Dispersion": "Between", "Dispersion_N6": "FPN"}
    pt = [age_test(data, v, COV_AGE, f"PT {names[k]}") for k, v in pairs.items()]
    _, q = author.fdr_correction([r["p"] for r in pt])
    for r, qq in zip(pt, q):
        r["p_fdr"] = qq
    for lc, ptc in pairs.items():
        data[lc + "_minus_PT"] = data[lc] - data[ptc]
        r = age_test(data, lc + "_minus_PT", COV_AGE, f"LC-PT {names[lc]}")
        r["F"] = r["t"] ** 2
        pt.append(r)
    tables["pt_control"] = pd.DataFrame(pt)

    print("\nValence and arousal ratings")
    ns = {}
    exec(notebook_source("valence arousal behaviour.ipynb"), ns)
    d = data.assign(Age=data.Age_Cat.map({"Young": "younger", "Old": "older"}))
    rat = []
    for cols, name in ((ns["VAL"], "Valence"), (ns["ARO"], "Arousal")):
        a = ns["mixed_anova"](d, cols)
        for eff in ("Condition", "Condition x Age"):
            print(f"  {name} {eff:16s} F({a[eff]['df1']}, {a[eff]['df2']}) = {a[eff]['F']:.2f}  p_GG = {a[eff]['p_GG']:.3g}")
            rat.append(dict(measure=name, test=eff, F=a[eff]["F"], df1=a[eff]["df1"],
                            df2=a[eff]["df2"], p=a[eff]["p"], p_GG=a[eff]["p_GG"]))
        for x, y, lab in ((cols[2], cols[0], "Negative vs Baseline"),
                          (cols[2], cols[1], "Negative vs Neutral"),
                          (cols[0], cols[1], "Baseline vs Neutral")):
            t, p, dz = ns["paired"](d, x, y)
            rat.append(dict(measure=name, test=lab, F=np.nan, df1=139, df2=np.nan, p=p,
                            p_GG=np.nan, t=t, dz=dz))
    tables["ratings"] = pd.DataFrame(rat)

    print("\nSaved reliability results (mean spatial correlation, G1 / G2 )")
    rel = []
    for kind, folder, fname, field in (
            ("Split-half", "Split_half", "splitHalf_reliability_combined_separate.mat", "splitHalf_results"),
            ("5-fold", "k_fold", "kfold_5_combined_separate.mat", "kfold_results")):
        for cond in ("negative_right", "negative_left", "neutral_right", "neutral_left"):
            m = scipy.io.loadmat(SENS / folder / cond / fname, squeeze_me=True, struct_as_record=False)[field]
            for grp in ("ALL", "YA", "OA"):
                arr = np.asarray(getattr(m, grp), dtype=float)      # (repeats[, folds], 3 gradients)
                vals = np.nanmean(arr.reshape(-1, arr.shape[-1]), axis=0)
                print(f"  {kind:10s} {cond:15s} {grp:3s}  {vals[0]:.3f}  {vals[1]:.3f}  {vals[2]:.3f}")
                for i, g in enumerate(("G1", "G2", "G3")):
                    rel.append(dict(kind=kind, condition=cond, group=grp, gradient=g, mean_rho=vals[i]))
    tables["reliability_saved"] = pd.DataFrame(rel)
    loo = []
    for c in [c for c in data.columns if c.endswith("_LOO")]:
        for g, sub in data.groupby("Age_Cat"):
            loo.append(dict(measure=c[:-4], group=g, mean=sub[c].mean(), sd=sub[c].std(),
                            min=sub[c].min(), max=sub[c].max()))
    tables["loo_summary"] = pd.DataFrame(loo)
    print("  LOO means:")
    print(pd.DataFrame(loo).pivot(index="measure", columns="group", values="mean").round(3).to_string())

    top = data.nlargest(14, "Global_Dispersion")[["Sub", "Age_Cat", "Global_Dispersion"]]
    tables["top14_global_dispersion"] = top.assign(rank=range(1, 15))
    print(f"\nTop 14 right global dispersion: {top.Age_Cat.value_counts().to_dict()}")
    return tables


def main():
    data = load_master()
    tables = run(data)
    OUTPUT.mkdir(exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(OUTPUT / f"{name}.csv", index=False, float_format="%.10g")
    print(f"\nCSV files written to {OUTPUT}")


if __name__ == "__main__":
    main()
