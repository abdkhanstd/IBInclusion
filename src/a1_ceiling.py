"""Q1: how large is faith-based self-exclusion?

The Findex barrier questions are asked of adults without an account at a financial institution (FI). For
country c and wave t:
  a_ct = share of ALL adults who lack an FI account and cite religious reasons (Findex "% age 15+" item);
         this is the ceiling C_ct: removing the faith barrier can raise FI account ownership by at most a_ct.
  u_ct = 1 - FI account ownership (Findex account indicator of the same wave);
  r_ct = a_ct / u_ct, the share of FI-unbanked adults citing religion (consistent base in both waves).
  r_item_ct = the published Findex share of the item's own base (the base implied by a_ct / r_item_ct exceeds
         u_ct in most country-waves, so r_item_ct <= r_ct; both are reported).
Muslim-specific component (an ecological decomposition, not a bound): r_ct = a_t + b_t M_c + e_ct across all
countries with barrier data; E_ct = min(max(b_t M_c, 0), r_ct). Country totals use adult population.
Uncertainty: nonparametric bootstrap over countries (re-estimating b_t), 9,999 draws.

Outputs: results/ceiling_country.csv, ceiling_summary.csv, ceiling_muslim_slope.csv, ceiling_slope_robust.csv
"""
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
BOOT = 9999


def load():
    b = pd.read_csv(os.path.join(DATA, "barriers.csv"))
    p = pd.read_csv(os.path.join(DATA, "panel.csv"))
    r = b[b.barrier == "religion"].drop(columns=["account"]).merge(
        p[["iso3", "wave", "fiaccount", "account"]], on=["iso3", "wave"], how="left")
    r["a"] = 100 * r.share_adults                       # % of all adults: FI-unbanked and citing religion
    r["u"] = 100 * (1 - r.fiaccount)                    # % of adults without an FI account
    r["r"] = 100 * r.share_adults / (1 - r.fiaccount)   # % of FI-unbanked citing religion
    r["r_item"] = 100 * r.share_unbanked                # as published (item-specific base)
    r["base_item"] = np.where(r.share_unbanked > 0, 100 * r.share_adults / r.share_unbanked, np.nan)  # % of adults
    r["mm"] = (r.muslim2020 >= 50).astype(int)
    return r.dropna(subset=["a", "u", "muslim"])


def decompose(g, b):
    exc = np.minimum(np.clip(b * g.muslim.values, 0, None), g.r.values)
    return exc


def totals(g, b):
    exc = decompose(g, b)
    w = g.pop_adult.values / 1e6
    unb = g.u.values / 100 * w
    return dict(C_adults_m=float(np.sum(g.a.values / 100 * w)), E_adults_m=float(np.sum(exc / 100 * unb)),
                unbanked_m=float(unb.sum()))


def main():
    os.makedirs(RES, exist_ok=True)
    r = load()
    slopes, rows = [], []
    r["E"] = np.nan
    rng = np.random.default_rng(0)
    for w in [2017, 2021]:
        g = r[r.wave == w]
        m = smf.ols("r ~ muslim", g).fit(cov_type="HC1")
        b = m.params.muslim
        slopes.append(dict(wave=w, intercept=m.params.Intercept, slope_per_10pp=10 * b, se_per_10pp=10 * m.bse.muslim,
                           p=m.pvalues.muslim, n=int(m.nobs)))
        r.loc[g.index, "E"] = decompose(g, b)
        # bootstrap over countries for b
        bs = []
        idx = np.arange(len(g))
        for _ in range(BOOT):
            gg = g.iloc[rng.choice(idx, len(idx))]
            X = np.column_stack([np.ones(len(gg)), gg.muslim.values])
            bs.append(np.linalg.lstsq(X, gg.r.values, rcond=None)[0][1])
        bs = np.array(bs)
        groups = {"EMDE": (g.emerging == 1), "EMDE, Muslim-majority": (g.emerging == 1) & (g.mm == 1),
                  "DEV, Muslim-majority": (g.developing == 1) & (g.mm == 1),
                  "EMDE, Muslim share < 10%": (g.emerging == 1) & (g.muslim2020 < 10),
                  "Muslim-majority, Islamic share < 15%": (g.mm == 1) & (g.ib_share < 15),
                  "Muslim-majority, Islamic share >= 15%": (g.mm == 1) & (g.ib_share >= 15)}
        for lab, msk in groups.items():
            gg = g[msk]
            if not len(gg):
                continue
            t = totals(gg, b)
            tb = np.array([totals(gg, bb)["E_adults_m"] for bb in bs[:2000]])
            wts = (gg.u * gg.pop_adult).values
            bi = gg.base_item.where(gg.base_item.notna(), gg.u)
            rows.append(dict(wave=w, group=lab, n=len(gg), r_median=gg.r.median(), r_mean=gg.r.mean(),
                             r_pw=np.average(gg.r, weights=wts), a_median=gg.a.median(),
                             r_item_median=gg.r_item.median(),
                             r_item_pw=np.average(gg.r_item, weights=(bi * gg.pop_adult).values),
                             C_adults_m=t["C_adults_m"], unbanked_m=t["unbanked_m"],
                             C_share_pw=100 * t["C_adults_m"] / t["unbanked_m"],
                             E_adults_m=t["E_adults_m"], E_lo=np.percentile(tb, 2.5), E_hi=np.percentile(tb, 97.5),
                             E_share_pw=100 * t["E_adults_m"] / t["unbanked_m"],
                             E_share_lo=100 * np.percentile(tb, 2.5) / t["unbanked_m"],
                             E_share_hi=100 * np.percentile(tb, 97.5) / t["unbanked_m"]))
    r.to_csv(os.path.join(RES, "ceiling_country.csv"), index=False)
    pd.DataFrame(slopes).to_csv(os.path.join(RES, "ceiling_muslim_slope.csv"), index=False)
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(RES, "ceiling_summary.csv"), index=False)

    # slope robustness
    rob = []
    specs = [("baseline", "r ~ muslim", None, "all"), ("+ log GDP per capita", "r ~ muslim + lgdppc", None, "all"),
             ("+ region effects", "r ~ muslim + C(region_geo)", None, "all"),
             ("+ region effects + log GDP per capita", "r ~ muslim + lgdppc + C(region_geo)", None, "all"),
             ("population weighted", "r ~ muslim", "pop_adult", "all"), ("EMDEs only", "r ~ muslim", None, "emde"),
             ("published item base", "r_item ~ muslim", None, "all")]
    for w in [2017, 2021]:
        gall = r[r.wave == w]
        mm = gall[(gall.emerging == 1) & (gall.mm == 1)]
        for lab, f, wt, samp in specs:
            # specifications with controls need GDP and region; the others use the headline sample
            g0 = gall.dropna(subset=["lgdppc", "region_geo"]) if ("lgdppc" in f or "region" in f) else gall
            g = g0[g0.emerging == 1] if samp == "emde" else g0
            m = (smf.wls(f, g, weights=g[wt]) if wt else smf.ols(f, g)).fit(cov_type="HC1")
            b = m.params["muslim"]
            yv = mm.r_item if "r_item" in f else mm.r
            exc = np.minimum(np.clip(b * mm.muslim, 0, None), yv)
            share = np.average(exc, weights=mm.u * mm.pop_adult)
            rob.append(dict(wave=w, spec=lab, slope_per_10pp=10 * b, se=10 * m.bse["muslim"], p=m.pvalues["muslim"],
                            n=int(m.nobs), E_share_pw=share))
    rob = pd.DataFrame(rob)
    rob.to_csv(os.path.join(RES, "ceiling_slope_robust.csv"), index=False)
    # implied base of the barrier items versus 1 - FI account (percentage points of adults)
    r["base_minus_u"] = r.base_item - r.u
    bd = r.dropna(subset=["base_minus_u"]).groupby("wave").agg(
        n=("base_minus_u", "size"), mean_diff=("base_minus_u", "mean"), median_diff=("base_minus_u", "median"),
        share_positive=("base_minus_u", lambda x: (x > 0).mean())).reset_index()
    bd.to_csv(os.path.join(RES, "ceiling_base_diff.csv"), index=False)
    print(bd.round(3))
    pd.set_option("display.width", 250)
    print(pd.DataFrame(slopes).round(3))
    print(s.round(2).to_string())
    print(rob.round(3).to_string())
    print(r[r.ib_share >= 50][["iso3", "wave", "ib_share", "r", "a", "u"]].round(2))


if __name__ == "__main__":
    main()
