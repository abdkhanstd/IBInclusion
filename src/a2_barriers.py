"""Q2: does Islamic banking reduce faith-based self-exclusion? Barrier-contrast design.

For country c and wave t in {2017, 2021}, a_ctb is the share of ALL adults who lack a financial-institution
account and cite barrier b (Findex "% age 15+" items). The primary outcome is the log contrast
  L_ct = ln(a_ct,rel + k) - (1/6) sum_{b != rel} ln(a_ctb + k),  k = 0.1 pp,
which removes exactly any factor common to all barriers of a country-wave (the unbanked share, its definition,
survey mode). The slope of dL_c on dIB_c (pp) is the faith-channel estimate. Secondary outcomes: the level
contrast in % of adults and in % of the unbanked (published item base). Placebo: the same log contrast for each
non-religious barrier against the other five non-religious barriers.

Sensitivity of the log contrast: offset k in {0.1, 0.5, 1, 2} pp (results/barrier_k.csv), and two measures that
need no offset: the religion share of all barrier citations, R_ct = a_ct,rel / sum_b a_ctb (first differences,
same inference), and a Poisson pseudo-likelihood model of a_ct,rel with country and wave effects and the log of
the summed non-religious citations as offset (cluster-robust and jackknife standard errors).

Outputs: results/barrier_estimates.csv, results/barrier_fd_data.csv, results/barrier_k.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FWLFit, ri_slope, holm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
BARS = ["distance", "cost", "documents", "trust", "religion", "funds", "family"]
OTHERS = [b for b in BARS if b != "religion"]
K = 0.1
BOUND_LOG = -np.log(0.75)   # equivalence bound: a 25 % relative change of religious exclusion per 10 pp


def wide(b, ibvar, k=K):
    a = b.pivot_table(index=["iso3", "wave"], columns="barrier", values="share_adults").reset_index()
    u = b.pivot_table(index=["iso3", "wave"], columns="barrier", values="share_unbanked").reset_index()
    a[BARS] = 100 * a[BARS]
    u[BARS] = 100 * u[BARS]
    meta = b.drop_duplicates(["iso3", "wave"])[["iso3", "wave", ibvar, "emerging", "developing", "ifsb_listed",
                                                "zero_ok", "lgdppc", "muslim", "muslim2020", "region_geo"]]
    w = a.merge(meta, on=["iso3", "wave"]).rename(columns={ibvar: "ib"})
    la = np.log(w[BARS] + k)
    w["L_rel"] = la["religion"] - la[OTHERS].mean(axis=1)
    w["A_rel"] = w["religion"] - w[OTHERS].mean(axis=1)
    w["R_rel"] = 100 * w["religion"] / w[BARS].sum(axis=1)
    w["sum_other"] = w[OTHERS].sum(axis=1)
    uu = u.set_index(["iso3", "wave"]).loc[list(zip(w.iso3, w.wave))].reset_index()
    w["U_rel"] = uu["religion"].values - uu[OTHERS].mean(axis=1).values
    for x in OTHERS:
        rest = [o for o in OTHERS if o != x]
        w[f"L_{x}"] = la[x] - la[rest].mean(axis=1)
    return w


def first_diff(w):
    cols = ["L_rel", "A_rel", "U_rel", "R_rel", "ib", "lgdppc"] + [f"L_{x}" for x in OTHERS]
    a = w[w.wave == 2017].set_index("iso3")
    c = w[w.wave == 2021].set_index("iso3")
    idx = a.index.intersection(c.index)
    d = (c.loc[idx, cols] - a.loc[idx, cols]).add_prefix("d_")
    for k in ["emerging", "developing", "ifsb_listed", "zero_ok", "region_geo", "muslim2020"]:
        d[k] = a.loc[idx, k]
    d["ib17"] = a.loc[idx, "ib"]
    d["rel17_adults"] = a.loc[idx, "religion"]
    d["R17"] = a.loc[idx, "R_rel"]
    return d.dropna(subset=["d_ib", "d_L_rel"]).reset_index()


def poisson_fe(w, keep):
    """Poisson pseudo-likelihood: religion citations (% of adults) on the Islamic share with country and wave
    effects and log(sum of non-religious citations) as offset; CR1 (country clusters) and jackknife SEs."""
    g = w[w.iso3.isin(keep) & w.ib.notna() & (w.sum_other > 0)].copy()
    g = g[g.groupby("iso3").iso3.transform("size") == 2]

    def est(gg):
        X = pd.get_dummies(gg[["iso3"]].astype(str), drop_first=True, dtype=float)
        X["w2021"] = (gg.wave == 2021).astype(float)
        X["ib"] = gg.ib / 10
        X = sm.add_constant(X)
        m = sm.GLM(gg.religion, X, family=sm.families.Poisson(), offset=np.log(gg.sum_other)).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(gg.iso3)[0]})
        return m.params["ib"], m.bse["ib"]
    b, se = est(g)
    jk = []
    for c in g.iso3.unique():
        jk.append(est(g[g.iso3 != c])[0])
    jk = np.array(jk)
    G = len(jk)
    se3 = np.sqrt((G - 1) / G * np.sum((jk - jk.mean()) ** 2))
    from scipy import stats
    return dict(beta=b, se=se, p=2 * stats.t.sf(abs(b / se), G - 1), se_cr3=se3,
                p_cr3=2 * stats.t.sf(abs(b / se3), G - 1), G=G, n=len(g))


def fit(d, y, name, sample, controls=(), bound=None):
    d = d.copy()
    f = FWLFit(d, y, "d_ib", fe=(), controls=controls, cluster="iso3", scale=10.0)
    out = f.summary(name=name, bound=bound)
    if not controls:
        out["p_ri"] = ri_slope(d[y], d["d_ib"], seed=0)[1]
    out.update(sample=sample, outcome=y, mean_dib=d.d_ib.mean(), sd_dib=d.d_ib.std(), mean_R17=d.R17.mean())
    return out


def main():
    b = pd.read_csv(os.path.join(DATA, "barriers.csv"))
    d = first_diff(wide(b, "ib_share"))
    dz = first_diff(wide(b, "ib_z"))
    di = first_diff(wide(b, "ib_interp"))
    dn = first_diff(wide(b, "ib_ns"))
    dp = first_diff(wide(b, "ib_np"))
    dx = first_diff(wide(b, "ib_sx"))
    S = {
        "EMDE": d[(d.emerging == 1) & (d.ifsb_listed == 1)],
        "DEV": d[(d.developing == 1) & (d.ifsb_listed == 1)],
        "EMDE, Muslim-majority": d[(d.emerging == 1) & (d.ifsb_listed == 1) & (d.muslim2020 >= 50)],
        "EMDE without Saudi Arabia": d[(d.emerging == 1) & (d.ifsb_listed == 1) & (d.iso3 != "SAU")],
        "ZERO": dz[(dz.emerging == 1) & ((dz.ifsb_listed == 1) | (dz.zero_ok == 1))],
        "EMDE, interpolated share": di[(di.emerging == 1) & (di.ifsb_listed == 1)],
        "EMDE, unmatched steps neutralised": dp[(dp.emerging == 1) & (dp.ifsb_listed == 1)],
        "EMDE, repeated values interpolated": dx[(dx.emerging == 1) & (dx.ifsb_listed == 1)],
        "EMDE, all large steps neutralised": dn[(dn.emerging == 1) & (dn.ifsb_listed == 1)],
    }
    os.makedirs(RES, exist_ok=True)
    pd.concat([v.assign(sample=k) for k, v in S.items()]).to_csv(os.path.join(RES, "barrier_fd_data.csv"),
                                                                 index=False)
    rows = []
    for s, dd in S.items():
        rows.append(fit(dd, "d_L_rel", "log contrast, % of adults", s, bound=BOUND_LOG))
        if s in ("EMDE", "DEV", "EMDE without Saudi Arabia"):
            rows.append(fit(dd, "d_R_rel", "religion share of all barrier citations (pp)", s))
        if s in ("EMDE", "DEV"):
            rows.append(fit(dd, "d_A_rel", "level contrast, % of adults", s))
            rows.append(fit(dd, "d_U_rel", "level contrast, % of unbanked (published base)", s))
            rows.append(fit(dd, "d_L_rel", "log contrast + change in log GDP pc", s, controls=("d_lgdppc",)))
        if s == "EMDE":
            for x in OTHERS:
                rows.append(fit(dd, f"d_L_{x}", f"placebo barrier: {x}", s))
    # interaction with the Muslim share (EMDE)
    e = S["EMDE"].copy()
    e["d_ib_x_m"] = e.d_ib * e.muslim2020 / 100
    f = FWLFit(e, "d_L_rel", "d_ib_x_m", fe=(), controls=("d_ib",), scale=10.0)
    r = f.summary(name="log contrast: Islamic share x Muslim share (effect at 100 % Muslim minus 0 %)")
    r.update(sample="EMDE", outcome="d_L_rel")
    rows.append(r)
    # Poisson pseudo-likelihood (no offset constant needed)
    w0 = wide(b, "ib_share")
    for s_, msk in [("EMDE", (S["EMDE"].iso3)), ("DEV", S["DEV"].iso3),
                    ("EMDE without Saudi Arabia", S["EMDE without Saudi Arabia"].iso3)]:
        r = poisson_fe(w0, set(msk))
        r.update(sample=s_, name="Poisson PML, religion citations (log points per 10 pp)", outcome="religion")
        rows.append(r)
    res = pd.DataFrame(rows)
    plac = res.name.str.startswith("placebo")
    res.loc[plac, "p_wcr_holm"] = holm(res.loc[plac, "p_wcr"])
    res.to_csv(os.path.join(RES, "barrier_estimates.csv"), index=False)
    pd.set_option("display.width", 260)
    cols = ["sample", "name", "beta", "se", "p", "p_wcr", "wb_lo", "wb_hi", "wb90_lo", "wb90_hi", "p_equiv", "p_ri",
            "p_cr3", "mde80", "lev_max", "lev_top", "G"]
    print(res[[c for c in cols if c in res]].round(3).to_string())
    m = S["EMDE"]
    print("mean religion % adults 2017 (EMDE IFSB):", round(m.rel17_adults.mean(), 2),
          "mean religion share of citations 2017:", round(m.R17.mean(), 2))
    # offset sensitivity
    krows = []
    for k in [0.1, 0.5, 1.0, 2.0]:
        dk = first_diff(wide(b, "ib_share", k))
        for s_, dd in [("EMDE", dk[(dk.emerging == 1) & (dk.ifsb_listed == 1)]),
                       ("EMDE without Saudi Arabia", dk[(dk.emerging == 1) & (dk.ifsb_listed == 1) & (dk.iso3 != "SAU")]),
                       ("DEV", dk[(dk.developing == 1) & (dk.ifsb_listed == 1)])]:
            r = fit(dd, "d_L_rel", f"log contrast, k = {k:g}", s_)
            r["k"] = k
            krows.append(r)
    kk = pd.DataFrame(krows)
    kk["pct"] = 100 * (np.exp(kk.beta) - 1)
    kk.to_csv(os.path.join(RES, "barrier_k.csv"), index=False)
    print(kk[["k", "sample", "beta", "pct", "p_wcr", "wb_lo", "wb_hi", "wb_nseg"]].round(3).to_string())


if __name__ == "__main__":
    main()
