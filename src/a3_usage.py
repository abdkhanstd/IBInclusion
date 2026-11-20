"""Q3: inclusion, usage and economic participation across Findex waves (2014-2024; 2011 as robustness).

Estimators for outcome y and Islamic banking share IB (pp of banking assets), coefficients per 10 pp of IB:
  B1 pooled OLS: wave FE + log GDP per capita + Muslim share
  B2 two-way FE: country FE + wave FE + log GDP per capita
  B3 B2 with region x wave FE
  D  matched contrast: (target - lambda * matched placebo) with country FE + wave FE + log GDP per capita;
     lambda = 1 in the main design, a grid of lambda as sensitivity analysis, and the two ends of the bracket
     for the shock-removing weight estimated in EMDEs without Islamic banking (D_wlo, D_whi).
Inference: CR1, CR3 (jackknife), CR2 with Satterthwaite degrees of freedom, effective number of clusters, and
the wild cluster restricted bootstrap (99,999 draws in every run) with confidence sets from the p-value function
on a grid (reported as a union of intervals when the set is not convex). Holm adjustment of the bootstrap p-values
within outcome families. 'excl90' is the largest absolute value in the 90 % bootstrap set: effects larger than it
are excluded by a 5 % equivalence test. 'mde80_wcr' is the minimum detectable effect (80 % power, 5 % two-sided)
implied by the width of the 95 % bootstrap set.

Loading bracket: with target T = l_T S + e_T and placebo P = l_P S + e_P, the contrast T - w P removes the
shock S when w = l_T / l_P. In EMDEs without Islamic banking, the within-country slope of P on T (lam_hat)
estimates l_P / l_T attenuated by the noise in T, and the slope of T on P (w_lo) estimates l_T / l_P attenuated
by the noise in P, so under classical measurement error w lies in [w_lo, w_hi] with w_hi = 1 / lam_hat
(w_lo = corr^2 / lam_hat). The lambda grid covers -0.5 to 5, and the breakdown file gives, for each pair, the
values of lambda in [-0.5, 10] at which the contrast is significant at 5 % (bootstrap).

Outputs: results/usage_estimates.csv, usage_loo.csv, contrast_lambda.csv, contrast_loading.csv,
contrast_breakdown.csv, twfe_weights.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FWLFit, holm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")

FINDEX = ["account", "fiaccount", "gap_gender", "gap_income", "save_formal", "borrow_formal", "inactive",
          "debitcard", "borrow_business", "borrow_family", "save_club", "borrow_health", "mobileaccount"]
FAMILIES = {
    "inclusion": ["account", "fiaccount", "gap_gender", "gap_income"],
    "usage": ["save_formal", "borrow_formal", "inactive", "debitcard"],
    "participation": ["borrow_business", "lfp_female", "selfemp"],
    "placebo": ["borrow_family", "save_club", "borrow_health", "mobileaccount"],
    "comparison": ["lfp_male"],
}
PAIRS = [("save_formal", "save_club"), ("borrow_formal", "borrow_family"), ("borrow_business", "borrow_health"),
         ("lfp_female", "lfp_male")]
LAMBDAS = [float(x) for x in np.round(np.arange(-0.5, 5.001, 0.25), 2)]
B = 99999
FULL = ("EMDE", "DEV", "EMDE without Saudi Arabia")
SPEC = {
    "B1": dict(fe=["wave"], controls=["lgdppc", "muslim"]),
    "B2": dict(fe=["iso3", "wave"], controls=["lgdppc"]),
    "B3": dict(fe=["iso3", ("region_geo", "wave")], controls=["lgdppc"]),
    "B2_noGDP": dict(fe=["iso3", "wave"], controls=[]),
}


def load():
    p = pd.read_csv(os.path.join(DATA, "panel.csv"))
    for c in FINDEX:
        p[c] = 100 * p[c]
    return p


def samples(p):
    base = p[p.wave >= 2014]
    emde = base[(base.emerging == 1) & base.ib_share.notna()]
    return {
        "EMDE": emde.assign(ib=lambda d: d.ib_share),
        "DEV": base[(base.developing == 1) & base.ib_share.notna()].assign(ib=lambda d: d.ib_share),
        "EMDE incl. 2011": p[(p.emerging == 1) & p.ib_share.notna()].assign(ib=lambda d: d.ib_share),
        "EMDE without Saudi Arabia": emde[emde.iso3 != "SAU"].assign(ib=lambda d: d.ib_share),
        "EXT": base[base.ib_share.notna()].assign(ib=lambda d: d.ib_share),
        "ZERO": base[(base.emerging == 1) & base.ib_z.notna()].assign(ib=lambda d: d.ib_z),
        "EMDE interpolated": base[(base.emerging == 1) & base.ib_interp.notna() & (base.ifsb_listed == 1)].assign(
            ib=lambda d: d.ib_interp),
        "EMDE systemic indicator": base[(base.emerging == 1) & base.ib_sys.notna()].assign(ib=lambda d: 10 * d.ib_sys),
        "EMDE, unmatched steps neutralised": emde.assign(ib=lambda d: d.ib_np),
        "EMDE, all large steps neutralised": emde.assign(ib=lambda d: d.ib_ns),
        "EMDE, repeated values interpolated": emde.assign(ib=lambda d: d.ib_sx),
    }


def est(d, y, spec, full):
    s = SPEC[spec]
    f = FWLFit(d, y, "ib", fe=s["fe"], controls=s["controls"], scale=10.0)
    r = f.summary(name=y, B=B, jack=full, ci90=full, cr2=full)
    if full:
        r["excl90"] = max(abs(r["wb90_lo"]), abs(r["wb90_hi"]))
    return r


def twfe_weights(d, y):
    """Weights of the two-way FE coefficient (de Chaisemartin and D'Haultfoeuille, 2020; country and wave
    effects, without the GDP control) on the cells with a positive Islamic share: w_ct proportional to eps_ct * IB_ct, where
    eps_ct is the residual of IB on the fixed effects."""
    from common import design, residualise
    d = d.dropna(subset=[y, "ib"])
    Z, _ = design(d, ["iso3", "wave"])
    eps = residualise(d.ib.values.astype(float), Z)
    pos = d.ib.values > 0
    w = eps[pos] * d.ib.values[pos]
    w = w / w.sum()
    return dict(outcome=y, n_cells=int(pos.sum()), share_negative=float((w < 0).mean()),
                sum_negative=float(w[w < 0].sum()))


def contrast(d, t, pl, lam):
    d = d.copy()
    d["_c"] = d[t] - lam * d[pl]
    return d


def loading(p):
    """Co-movement of target and placebo in EMDEs without Islamic banking (zero_ok), 2014-2024: slope of the
    within-country (country and wave FE, log GDP per capita) placebo on the within-country target, with a
    country bootstrap."""
    z = p[(p.zero_ok == 1) & (p.emerging == 1) & (p.wave >= 2014)]
    out = []
    rng = np.random.default_rng(0)
    for t, pl in PAIRS:
        d = z.dropna(subset=[t, pl, "lgdppc"])
        f = FWLFit(d, pl, t, fe=["iso3", "wave"], controls=["lgdppc"])
        fr = FWLFit(d, t, pl, fe=["iso3", "wave"], controls=["lgdppc"])
        corr = np.corrcoef(f.xt, f.yt)[0, 1]
        cs = d.iso3.unique()
        bs = []
        for _ in range(999):
            pick = rng.choice(cs, len(cs))
            dd = pd.concat([d[d.iso3 == c].assign(iso3=f"{c}_{i}") for i, c in enumerate(pick)])
            try:
                bs.append(FWLFit(dd, pl, t, fe=["iso3", "wave"], controls=["lgdppc"]).beta)
            except Exception:
                pass
        out.append(dict(target=t, placebo=pl, lam_hat=f.beta, lam_lo=np.percentile(bs, 2.5),
                        lam_hi=np.percentile(bs, 97.5), corr=corr, w_lo=fr.beta, w_hi=1 / f.beta, n=f.n, G=f.G))
    return pd.DataFrame(out)


def main():
    p = load()
    S = samples(p)
    os.makedirs(RES, exist_ok=True)
    load_df = loading(p)
    load_df.to_csv(os.path.join(RES, "contrast_loading.csv"), index=False)
    w_lo = dict(zip(load_df.target, load_df.w_lo))
    w_hi = dict(zip(load_df.target, load_df.w_hi))
    rows = []
    outcomes = sum(FAMILIES.values(), [])
    for sname, d in S.items():
        print("sample", sname, flush=True)
        full = sname in FULL
        ests = ["B1", "B2", "B3", "B2_noGDP"] if sname in ("EMDE", "DEV") else ["B2"]
        for y in outcomes:
            fam = [k for k, v in FAMILIES.items() if y in v][0]
            for e in ests:
                if d[y].notna().sum() < 20:
                    continue
                r = est(d, y, e, full and e != "B2_noGDP")
                r.update(sample=sname, estimator=e, family=fam, outcome=y)
                rows.append(r)
        for t, pl in PAIRS:
            for lab, lam in [("D", 1.0), ("D_wlo", w_lo[t]), ("D_whi", w_hi[t])]:
                if lab != "D" and not full:
                    continue
                dd = contrast(d, t, pl, lam)
                f = FWLFit(dd, "_c", "ib", fe=["iso3", "wave"], controls=["lgdppc"], scale=10.0)
                r = f.summary(name=f"{t}-{lam:.2f}*{pl}", B=B, jack=full, ci90=full, cr2=full)
                if full:
                    r["excl90"] = max(abs(r["wb90_lo"]), abs(r["wb90_hi"]))
                r.update(sample=sname, estimator=lab, family="contrast", outcome=f"{t}__minus__{pl}", target=t,
                         placebo=pl, lam=lam)
                rows.append(r)
            if sname in ("EMDE", "DEV"):
                dd = contrast(d, t, pl, 1.0)
                f = FWLFit(dd, "_c", "ib", fe=["wave"], controls=["lgdppc", "muslim"], scale=10.0)
                r = f.summary(name="pooled contrast", B=B, jack=False)
                r.update(sample=sname, estimator="D_pooled", family="contrast", outcome=f"{t}__minus__{pl}",
                         target=t, placebo=pl, lam=1.0)
                rows.append(r)
    res = pd.DataFrame(rows)
    for (s, e, fam), g in res.groupby(["sample", "estimator", "family"]):
        if g.p_wcr.notna().all():
            res.loc[g.index, "p_wcr_holm"] = holm(g.p_wcr.values)
    res.to_csv(os.path.join(RES, "usage_estimates.csv"), index=False)

    print("lambda grid", flush=True)
    lam_rows = []
    for sname in ["EMDE", "DEV"]:
        d = S[sname]
        for t, pl in PAIRS:
            for lam in LAMBDAS:
                dd = contrast(d, t, pl, lam)
                f = FWLFit(dd, "_c", "ib", fe=["iso3", "wave"], controls=["lgdppc"], scale=10.0)
                r = f.summary(B=B, jack=False)
                lam_rows.append(dict(sample=sname, target=t, placebo=pl, lam=lam, beta=r["beta"], wb_lo=r["wb_lo"],
                                     wb_hi=r["wb_hi"], wb_nseg=r["wb_nseg"], wb_set=r["wb_set"], p_wcr=r["p_wcr"]))
    pd.DataFrame(lam_rows).to_csv(os.path.join(RES, "contrast_lambda.csv"), index=False)

    # breakdown: values of lambda at which the contrast is significant at 5 % (bootstrap p-value only)
    print("breakdown", flush=True)
    bd = []
    grid = np.round(np.arange(-0.5, 10.001, 0.1), 2)
    for sname in ["EMDE", "DEV", "EMDE without Saudi Arabia"]:
        d = S[sname]
        for t, pl in PAIRS:
            sig = []
            for lam in grid:
                dd = contrast(d, t, pl, lam)
                f = FWLFit(dd, "_c", "ib", fe=["iso3", "wave"], controls=["lgdppc"], scale=10.0)
                wG = f.weights(B, 0)
                if f.wcr_p(0.0, wG) < 0.05:
                    sig.append((lam, f.beta * 10))
            row = load_df[load_df.target == t].iloc[0]
            inb = [l for l, _ in sig if row.w_lo - 1e-9 <= l <= row.w_hi + 1e-9]
            bd.append(dict(sample=sname, target=t, placebo=pl, w_lo=row.w_lo, w_hi=row.w_hi, n_sig=len(sig),
                           n_sig_in_bracket=len(inb),
                           sig_lams=" ".join(f"{l:g}" for l, _ in sig),
                           min_sig_lam=min((l for l, _ in sig), default=np.nan),
                           max_sig_lam=max((l for l, _ in sig), default=np.nan)))
    pd.DataFrame(bd).to_csv(os.path.join(RES, "contrast_breakdown.csv"), index=False)
    print(pd.DataFrame(bd).round(2).to_string())

    tw = []
    for sname in ["EMDE", "DEV"]:
        for y in ["account", "save_formal", "borrow_formal"]:
            r = twfe_weights(S[sname], y)
            r["sample"] = sname
            tw.append(r)
    pd.DataFrame(tw).to_csv(os.path.join(RES, "twfe_weights.csv"), index=False)
    print(pd.DataFrame(tw).round(3))

    print("leave-one-out", flush=True)
    loo = []
    for sname in ["EMDE", "DEV"]:
        d = S[sname]
        for c in sorted(d.iso3.unique()):
            dd = d[d.iso3 != c]
            for y in outcomes:
                if dd[y].notna().sum() < 20:
                    continue
                f = FWLFit(dd, y, "ib", fe=["iso3", "wave"], controls=["lgdppc"], scale=10.0)
                loo.append(dict(sample=sname, dropped=c, outcome=y, estimator="B2", beta=f.beta * 10, p=f.p))
            for t, pl in PAIRS:
                d2 = contrast(dd, t, pl, 1.0)
                f = FWLFit(d2, "_c", "ib", fe=["iso3", "wave"], controls=["lgdppc"], scale=10.0)
                loo.append(dict(sample=sname, dropped=c, outcome=f"{t}__minus__{pl}", estimator="D", beta=f.beta * 10,
                                p=f.p))
    pd.DataFrame(loo).to_csv(os.path.join(RES, "usage_loo.csv"), index=False)
    pd.set_option("display.width", 260)
    print(load_df.round(3))
    for s in ["EMDE", "DEV"]:
        m = res[(res["sample"] == s) & res.estimator.isin(["B1", "B2", "D", "D_wlo", "D_whi"])]
        print(m[["estimator", "family", "outcome", "beta", "p_wcr", "p_wcr_holm", "wb_lo", "wb_hi", "wb_nseg", "excl90",
                 "p_cr3", "p_cr2", "df_cr2", "g_star", "lev_max", "lev_top", "G"]].round(3).to_string())


if __name__ == "__main__":
    main()
