"""Faith channel on the four-wave barrier panel (2011, 2014, 2017, 2021/22), from the codebook frequencies.

s_ctb is the share of the respondents asked (adults without a financial-institution account) who cite reason b
(data/barriers_panel.csv). Measures:
  L_ct = ln(s_ct,rel + k) - (1/6) sum_{b != rel} ln(s_ctb + k), k = 0.5 pp of the asked (log religion contrast)
  R_ct = 100 s_ct,rel / sum_b s_ctb                                  (religion share of all citations, no offset)
  Poisson pseudo-likelihood of the religion yes-count with country and wave effects and the log of the summed
  yes-counts of the six other reasons as offset (no offset constant).
Main design: L_ct = a_c + d_t + beta IB_ct + e_ct on the waves 2014, 2017 and 2021 (the 2011 survey precedes the
first IFSB edition and enters only as a robustness check), countries with at least two waves.

Implied inclusion (accounting model, Proposition 3): if each 10 pp of share brings a fraction theta of the
faith-excluded into the system, the log contrast falls by ln(1 - theta) - mean_b ln(1 - q_b theta s_rel/s_b), where
q_b is the fraction of them who also cite b. theta lies between the solutions for q_b = 0 and q_b = 1, and the
account gain per 10 pp is theta times the share of adults who lack an FI account and cite religion.

Dose test (Proposition 3): FI account ownership (pp) on IB_ct/10 x pool_c with IB_ct/10 as control and country and
wave effects, where pool_c is the share of adults who lacked an FI account and cited religion in 2011 (before the
first IFSB edition); the coefficient estimates theta directly.

Falsification: the change in L from 2011 to 2014 regressed on the later change in the share from 2014 to 2021.
Efficiency check: feasible GLS with weights 1 / (sampling variance of L_ct + country-wave variance).
Pre-trends and interview mode: a Saudi-specific trend, country-specific trends (2011-2021), the waves in the
2017 interview mode only, and interview-mode indicators.

Muslim account gap: share of the income-conditional Muslim gap in FI account ownership that religious
self-exclusion can explain (results/muslim_gap.csv).

Outputs: results/barrier_panel_estimates.csv, results/barrier_panel_k.csv, results/bound_waves.csv,
results/implied_gain.csv, results/dose_estimates.csv, results/barrier_panel_data.csv, results/muslim_gap.csv,
results/zero_trend.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import FWLFit, design, holm, residualise, ri_slope  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
BARS = ["distance", "cost", "documents", "trust", "religion", "funds", "family"]
OTHERS = [b for b in BARS if b != "religion"]
K = 0.5
BOUND_LOG = -np.log(0.75)
MAIN_WAVES = (2014, 2017, 2021)
IBVARS = {"ib_share": "as published", "ib_interp": "interpolated", "ib_np": "unmatched steps neutralised",
          "ib_ns": "all large steps neutralised", "ib_sx": "repeated values interpolated", "ib_z": "ZERO"}


def measures(d, k=K):
    d = d.copy()
    S = d[[f"s_{b}" for b in BARS]]
    la = np.log(S + k)
    oth = [f"s_{b}" for b in OTHERS]
    d["L_rel"] = la["s_religion"] - la[oth].mean(axis=1)
    d["R_rel"] = 100 * d.s_religion / S.sum(axis=1)
    for x in OTHERS:
        rest = [f"s_{o}" for o in OTHERS if o != x]
        d[f"L_{x}"] = la[f"s_{x}"] - la[rest].mean(axis=1)
    d["y_other"] = d[[f"y_{b}" for b in OTHERS]].sum(axis=1)
    return d


def sample(d, ibvar="ib_share", waves=MAIN_WAVES, flag="emerging", extra=None):
    s = d[d.wave.isin(waves) & (d[flag] == 1)].copy()
    if ibvar == "ib_z":
        s = s[(s.ifsb_listed == 1) | (s.zero_ok == 1)]
    else:
        s = s[s.ifsb_listed == 1]
    s["ib"] = s[ibvar]
    s = s[s.ib.notna()]
    if extra is not None:
        s = extra(s)
    return s[s.groupby("iso3").iso3.transform("size") >= 2]


def fit(s, y, name, label, bound=None):
    f = FWLFit(s, y, "ib", fe=("iso3", "wave"), cluster="iso3", scale=10.0)
    out = f.summary(name=name, bound=bound, cr2=True)
    out.update(sample=label, outcome=y, waves=int(s.wave.nunique()), mean_R=s.R_rel.mean(),
               sd_dib=s.groupby("iso3").ib.transform(lambda v: v - v.mean()).std())
    return out


def poisson(s, label):
    g = s[(s.y_other > 0)].copy()
    g = g[g.groupby("iso3").iso3.transform("size") >= 2]

    def est(gg):
        X = pd.get_dummies(gg[["iso3", "wave"]].astype(str), drop_first=True, dtype=float)
        X["ib"] = gg.ib.values / 10
        X = sm.add_constant(X)
        m = sm.GLM(gg.y_religion.values, X, family=sm.families.Poisson(), offset=np.log(gg.y_other.values)).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(gg.iso3)[0]})
        return m.params["ib"], m.bse["ib"]
    b, se = est(g)
    jk = np.array([est(g[g.iso3 != c])[0] for c in g.iso3.unique()])
    G = len(jk)
    se3 = np.sqrt((G - 1) / G * np.sum((jk - jk.mean()) ** 2))
    q = stats.t.ppf(0.975, G - 1)
    return dict(name="Poisson PML, religion citations (log points per 10 pp)", sample=label, outcome="y_religion",
                beta=b, se=se, p=2 * stats.t.sf(abs(b / se), G - 1), ci_lo=b - q * max(se, se3),
                ci_hi=b + q * max(se, se3), se_cr3=se3, p_cr3=2 * stats.t.sf(abs(b / se3), G - 1), G=G, n=len(g))


def long_diff(s, y="L_rel", a=2014, b=2021):
    w = s.pivot_table(index="iso3", columns="wave", values=[y, "ib"])
    w = w.dropna(subset=[(y, a), (y, b), ("ib", a), ("ib", b)])
    dy = w[(y, b)] - w[(y, a)]
    dx = (w[("ib", b)] - w[("ib", a)]) / 10
    beta, p = ri_slope(dy.values, dx.values, seed=0)
    return dict(beta_ld=beta, p_ri_ld=p, n_ld=len(w))


def sampling_var(s, k=K):
    """Delta-method sampling variance of L_ct from binomial counts (covariances between reasons ignored)."""
    n = s.n_asked.values
    pr = s.s_religion.values / 100
    v = pr * (1 - pr) / n * 100 ** 2 / (s.s_religion.values + k) ** 2
    for b in OTHERS:
        q = s[f"s_{b}"].values / 100
        v = v + (1 / 36) * q * (1 - q) / n * 100 ** 2 / (s[f"s_{b}"].values + k) ** 2
    return v


class WFit(FWLFit):
    """FWLFit with observation weights (all inference on the weighted, residualised data)."""

    def __init__(self, df, y, x, w, fe=(), cluster="iso3", scale=1.0):
        d = df.dropna(subset=[y, x, w]).reset_index(drop=True)
        sw = np.sqrt(d[w].values.astype(float))
        self.df, self.y, self.x, self.fe, self.controls, self.w = d, y, x, fe, [], w
        self.n = len(d)
        Z, _ = design(d, fe, ())
        Zw = Z * sw[:, None]
        self.k = common.rank(Zw) + 1
        self.yt = residualise(d[y].values.astype(float) * sw, Zw)
        self.xt = residualise(d[x].values.astype(float) * sw, Zw)
        self.cluster = cluster
        self.groups = pd.factorize(d[cluster])[0]
        self.G = int(self.groups.max() + 1)
        self.scale = scale
        if any(isinstance(f, str) and f == cluster for f in fe):
            self.k -= self.G - 1
        sxx = self.xt @ self.xt
        self.beta = (self.xt @ self.yt) / sxx
        self.resid = self.yt - self.xt * self.beta
        self.adj = (self.G / (self.G - 1)) * ((self.n - 1) / max(self.n - self.k, 1))
        sc = np.bincount(self.groups, weights=self.xt * self.resid, minlength=self.G)
        self.se = np.sqrt(self.adj * (sc @ sc)) / sxx
        self.t = self.beta / self.se if self.se > 0 else np.nan
        self.p = 2 * stats.t.sf(abs(self.t), self.G - 1)
        self.A = np.bincount(self.groups, weights=self.xt * self.yt, minlength=self.G)
        self.C = np.bincount(self.groups, weights=self.xt * self.xt, minlength=self.G)

    def jackknife(self):
        bs = []
        for g in np.unique(self.df[self.cluster]):
            sub = self.df[self.df[self.cluster] != g]
            bs.append(WFit(sub, self.y, self.x, self.w, self.fe, self.cluster).beta)
        bs = np.array(bs)
        G = len(bs)
        se = np.sqrt((G - 1) / G * np.sum((bs - bs.mean()) ** 2))
        return se, (2 * stats.t.sf(abs(self.beta / se), self.G - 1) if se > 0 else np.nan)


def fit_wls(s, label):
    """Feasible GLS: weights 1 / (v_ct + sigma2_nu), v_ct the sampling variance of L_ct and sigma2_nu the
    country-wave variance beyond sampling noise (residual variance of the unweighted fit with fixed-effect degrees
    of freedom minus the mean sampling variance)."""
    s = s.copy().reset_index(drop=True)
    v = sampling_var(s)
    f0 = FWLFit(s, "L_rel", "ib", fe=("iso3", "wave"), cluster="iso3", scale=10.0)
    dof = f0.n - (f0.G + s.wave.nunique())
    sig2 = max((f0.resid @ f0.resid) / dof - v.mean(), 0.0)
    s["w_prec"] = 1 / (v + sig2)
    f = WFit(s, "L_rel", "ib", "w_prec", fe=("iso3", "wave"), scale=10.0)
    out = f.summary(name="log contrast, precision-weighted", bound=BOUND_LOG)
    out.update(sample=label, outcome="L_rel", sigma2_nu=sig2, mean_sampling_var=v.mean())
    return out


def fit_controls(s, label, name, controls):
    f = FWLFit(s, "L_rel", "ib", fe=("iso3", "wave"), controls=tuple(controls), cluster="iso3", scale=10.0)
    out = f.summary(name=name, bound=BOUND_LOG, cr2=True)
    out.update(sample=label, outcome="L_rel", n_controls=len(controls))
    return out


def trend_variants(d):
    """Pre-trend adjustments: a Saudi-specific linear trend (waves 2014-2021) and country-specific linear trends
    for countries observed in at least three of the four waves 2011-2021."""
    out = []
    s = sample(d).copy()
    s["trend_SAU"] = np.where(s.iso3 == "SAU", s.wave - 2017, 0.0)
    out.append(fit_controls(s, "EMDE", "log contrast, Saudi-specific trend", ["trend_SAU"]))
    for lab, ibv, flag in [("EMDE", "ib_share", "emerging"), ("EMDE, unmatched steps neutralised", "ib_np", "emerging"),
                           ("DEV", "ib_share", "developing")]:
        s = sample(d, ibv, (2011,) + MAIN_WAVES, flag)
        s = s[s.groupby("iso3").iso3.transform("size") >= 3].copy()
        ctr = []
        for c in sorted(s.iso3.unique())[1:]:
            s[f"tr_{c}"] = np.where(s.iso3 == c, s.wave - 2016, 0.0)
            ctr.append(f"tr_{c}")
        # common linear trend is absorbed by the wave effects; one country is the reference
        out.append(fit_controls(s, lab, "log contrast, country-specific trends (2011-2021)", ctr))
    return out


def mode_variants(d):
    """Interview mode: (a) keep only the waves of each country interviewed in the same mode as in 2017 (face or
    phone; 'other' dropped); (b) indicators for phone and unspecified modes as controls."""
    out = []
    s = sample(d).copy()
    ref = s[s.wave == 2017].set_index("iso3").mode_class
    s["ref_mode"] = s.iso3.map(ref)
    k = s[(s.mode_class == s.ref_mode) & (s.mode_class != "other")]
    k = k[k.groupby("iso3").iso3.transform("size") >= 2]
    r = fit(k, "L_rel", "log contrast, same interview mode as in 2017", "EMDE", bound=BOUND_LOG)
    out.append(r)
    s["phone"] = (s.mode_class == "phone").astype(float)
    s["mode_other"] = (s.mode_class == "other").astype(float)
    out.append(fit_controls(s, "EMDE", "log contrast, interview-mode indicators", ["phone", "mode_other"]))
    return out


def lead_test(d):
    """Falsification: the change in the log contrast from 2011 to 2014 (before most of the share changes) regressed
    on the later change in the share from 2014 to 2021. Under parallel trends the slope is zero."""
    out = []
    for lab, ibv, flag, drop in [("EMDE", "ib_share", "emerging", None),
                                 ("EMDE, unmatched steps neutralised", "ib_np", "emerging", None),
                                 ("EMDE without Saudi Arabia", "ib_share", "emerging", "SAU"),
                                 ("DEV", "ib_share", "developing", None)]:
        e = d[(d[flag] == 1) & (d.ifsb_listed == 1)]
        if drop:
            e = e[e.iso3 != drop]
        w = e.pivot_table(index="iso3", columns="wave", values=["L_rel", ibv])
        x = pd.DataFrame({"dL_pre": w[("L_rel", 2014)] - w[("L_rel", 2011)],
                          "dIB_post": w[(ibv, 2021)] - w[(ibv, 2014)]}).dropna().reset_index()
        f = FWLFit(x, "dL_pre", "dIB_post", fe=(), cluster="iso3", scale=10.0)
        o = f.summary(name="lead test: change 2011-2014 on share change 2014-2021", cr2=True)
        o.update(sample=lab, outcome="dL_pre", p_ri=ri_slope(x.dL_pre.values, x.dIB_post.values / 10, seed=0)[1])
        out.append(o)
    return out


def zero_trend(d):
    """Mean change of the log religion contrast from 2014 to 2021 in the ZERO comparison economies (no Islamic
    banking, Muslim share below 1%) and in the IFSB-listed EMDEs, countries observed in both waves."""
    rows = []
    for lab, msk in [("ZERO comparison economies", (d.zero_ok == 1)),
                     ("IFSB-listed EMDEs", (d.ifsb_listed == 1) & d.ib_share.notna())]:
        g = d[(d.emerging == 1) & msk & d.wave.isin([2014, 2021])]
        w = g.pivot_table(index="iso3", columns="wave", values=["L_rel", "fiaccount"]).dropna()
        rows.append(dict(group=lab, n=len(w), d_L=float((w[("L_rel", 2021)] - w[("L_rel", 2014)]).mean()),
                         d_fiaccount_pp=float(100 * (w[("fiaccount", 2021)] - w[("fiaccount", 2014)]).mean())))
    return pd.DataFrame(rows)


def muslim_gap(d, B=9999):
    """Share of the Muslim account gap that faith-based exclusion can explain. Across countries with codebook barrier
    data, wave by wave: FI account ownership (pp) and the share of adults who lack an FI account and cite a reason
    (pp) on the Muslim population share (0-1) and log GDP per capita (OLS, HC1). The ratio of the two Muslim-share
    coefficients, -b_reason / b_account, is an upper bound on the part of the income-conditional Muslim account gap
    that the reason can explain if every citation reflected a binding barrier. 95% interval: percentile bootstrap over
    countries (draws with a non-negative account gap discarded). Specifications: all countries (baseline), EMDE, DEV,
    with region effects, weighted by adult population; benchmarks for each of the six other reasons."""
    rows = []
    specs = [("all countries", "religion", lambda g: g, None, False),
             ("EMDE", "religion", lambda g: g[g.emerging == 1], None, False),
             ("DEV", "religion", lambda g: g[g.developing == 1], None, False),
             ("all countries, region effects", "religion", lambda g: g, "region_geo", False),
             ("all countries, adult-population weights", "religion", lambda g: g, None, True),
             ("all countries, benchmark: insufficient funds", "funds", lambda g: g, None, False),
             ("all countries, benchmark: family has one", "family", lambda g: g, None, False),
             ("all countries, benchmark: too far away", "distance", lambda g: g, None, False),
             ("all countries, benchmark: too expensive", "cost", lambda g: g, None, False),
             ("all countries, benchmark: documentation", "documents", lambda g: g, None, False),
             ("all countries, benchmark: lack of trust", "trust", lambda g: g, None, False)]
    for wave in (2011, 2014, 2017, 2021):
        base = d[(d.wave == wave)].dropna(subset=["fiaccount", "lgdppc", "muslim", "rel_adults"])
        for lab, reason, sel, fe, weighted in specs:
            g = sel(base).copy()
            cols = [g.muslim.values / 100, g.lgdppc.values]
            if fe:
                dm = pd.get_dummies(g[fe].astype(str), drop_first=True, dtype=float).values
                X = sm.add_constant(np.column_stack(cols + [dm]))
            else:
                X = sm.add_constant(np.column_stack(cols))
            ya = 100 * g.fiaccount.values
            yr = g.fi_unbanked.values * g[f"s_{reason}"].values / 100
            w = g.pop_adult.values if weighted else np.ones(len(g))
            ma = sm.WLS(ya, X, weights=w).fit(cov_type="HC1")
            mr = sm.WLS(yr, X, weights=w).fit(cov_type="HC1")
            rng = np.random.default_rng(0)
            n = len(g)
            sw = np.sqrt(w)
            ratios = []
            for _ in range(B):
                i = rng.integers(0, n, n)
                Xi = X[i] * sw[i, None]
                ba = np.linalg.lstsq(Xi, ya[i] * sw[i], rcond=None)[0][1]
                br = np.linalg.lstsq(Xi, yr[i] * sw[i], rcond=None)[0][1]
                ratios.append(-br / ba if ba < 0 else np.nan)
            ratios = np.array(ratios)
            rows.append(dict(wave=wave, spec=lab, reason=reason, n=n, gap_account=ma.params[1],
                             gap_account_se=ma.bse[1], gap_account_p=ma.pvalues[1], gap_faith=mr.params[1],
                             gap_faith_se=mr.bse[1], gap_faith_p=mr.pvalues[1], share=-mr.params[1] / ma.params[1],
                             share_lo=np.nanpercentile(ratios, 2.5), share_hi=np.nanpercentile(ratios, 97.5),
                             share_na=float(np.isnan(ratios).mean())))
    return pd.DataFrame(rows)


def bound_waves(d):
    rows = []
    for wave in (2011, 2014, 2017, 2021):
        w = d[(d.wave == wave) & d.muslim2020.notna()]
        mm = w[(w.emerging == 1) & (w.muslim2020 >= 50)]
        unb = mm.pop_adult * mm.fi_unbanked / 100
        # Muslim-share slope across all countries (pp of the asked per 10 pp of Muslim share), HC1
        X = sm.add_constant(w.muslim2020.values / 10)
        m = sm.OLS(w.s_religion.values, X).fit(cov_type="HC1")
        bt = m.params[1]
        comp = np.minimum(np.maximum(bt * mm.muslim2020.values / 10, 0), mm.s_religion.values)
        rng = np.random.default_rng(0)
        bs = []
        n = len(w)
        for _ in range(9999):
            i = rng.integers(0, n, n)
            Xb = X[i]
            yb = w.s_religion.values[i]
            bb = np.linalg.lstsq(Xb, yb, rcond=None)[0][1]
            cb = np.minimum(np.maximum(bb * mm.muslim2020.values / 10, 0), mm.s_religion.values)
            bs.append(np.average(cb, weights=unb))
        rows.append(dict(wave=wave, n_mm=len(mm), n_all=len(w), median=mm.s_religion.median(),
                         weighted=np.average(mm.s_religion, weights=unb), unbanked_m=unb.sum() / 1e6,
                         religion_m=(unb * mm.s_religion / 100).sum() / 1e6,
                         low_muslim=np.average(w[(w.emerging == 1) & (w.muslim2020 < 10)].s_religion,
                                               weights=(w.pop_adult * w.fi_unbanked)[(w.emerging == 1) &
                                                                                      (w.muslim2020 < 10)]),
                         slope=bt, slope_se=m.bse[1], slope_p=m.pvalues[1],
                         component=np.average(comp, weights=unb), comp_lo=np.percentile(bs, 2.5),
                         comp_hi=np.percentile(bs, 97.5),
                         rel_adults_w=np.average(mm.rel_adults, weights=mm.pop_adult),
                         least_cited=np.mean(mm[[f"s_{b}" for b in OTHERS]].min(axis=1) >= mm.s_religion),
                         # Proposition 2 with q_b = 1: small faith-channel inclusion lowers L if mean_b s_rel/s_b < 1
                         prop2_cond=np.mean(np.nanmean(np.column_stack(
                             [mm.s_religion / mm[f"s_{b}"].replace(0, np.nan) for b in OTHERS]), axis=1) < 1),
                         n_prop2=len(mm)))
    return pd.DataFrame(rows)


def theta_from_beta(beta, rho):
    """theta for q_b = 0 (closed form) and q_b = 1 (numerical) given a log-contrast change beta per 10 pp and the
    ratios rho_b = s_rel / s_b of the baseline shares."""
    t0 = 1 - np.exp(beta)

    def f(t):
        return np.log(1 - t) - np.mean(np.log(1 - t * rho)) - beta
    hi = min(0.999, 0.999 / max(rho.max(), 1e-9))
    try:
        t1 = optimize.brentq(f, -20, hi)
    except ValueError:
        t1 = np.nan
    return t0, t1


def access_shift(sample_name, u, a_rel):
    """Largest simultaneous access-channel inclusion not excluded by the account data: the upper end of the 90% WCR
    set of the two-way FE effect on FI account ownership (pp per 10 pp, results/usage_estimates.csv, same sample,
    EMDE where the sample has no set). If access entrants cite the other barriers in proportion, it raises L by about
    x_A / (u - a_rel) (first order), so faith-channel reach is computed at beta - shift."""
    path = os.path.join(RES, "usage_estimates.csv")
    if not os.path.exists(path):
        return np.nan, np.nan
    us = pd.read_csv(path)
    us = us[(us.estimator == "B2") & (us.outcome == "fiaccount")]
    r = us[us["sample"] == sample_name]
    if not len(r) or np.isnan(r.wb90_hi.iloc[0]):
        r = us[us["sample"] == "EMDE"]
    x_a = max(float(r.wb90_hi.iloc[0]), 0.0)
    return x_a, x_a / (u - a_rel)


def implied_gain(res, d):
    base = d[(d.wave == 2014) & (d.emerging == 1) & (d.muslim2020 >= 50)]
    w = base.pop_adult * base.fi_unbanked
    s = {b: np.average(base[f"s_{b}"], weights=w) for b in BARS}
    rho = np.array([s["religion"] / s[b] for b in OTHERS])
    pool = np.average(base.rel_adults, weights=base.pop_adult)
    u = np.average(base.fi_unbanked, weights=base.pop_adult)
    rows = []
    for _, r in res[res.name.isin(["log contrast", "log contrast, precision-weighted"])].iterrows():
        x_a, shift = access_shift(r["sample"], u, pool)
        for what, b in [("point", r.beta), ("set lower end", r.wb_lo), ("set upper end", r.wb_hi)]:
            t0, t1 = theta_from_beta(b, rho)
            a0, a1 = theta_from_beta(b - shift, rho) if not np.isnan(shift) else (np.nan, np.nan)
            rows.append(dict(sample=r["sample"], estimator=r["name"], what=what, beta=b, theta_q0=t0, theta_q1=t1,
                             gain_q0=t0 * pool, gain_q1=t1 * pool, pool=pool, s_rel=s["religion"], u=u,
                             access_xa=x_a, access_shift=shift, gain_q0_access=a0 * pool, gain_q1_access=a1 * pool))
    return pd.DataFrame(rows)


def dose(d):
    """FI account ownership (pp) on IB/10 x pool (pool: % of adults without an FI account citing religion,
    2011 wave, or 2014 as an alternative), with IB/10, country and wave effects."""
    p = pd.read_csv(os.path.join(DATA, "panel.csv"))
    rows = []
    for pw in (2011, 2014):
        pool = d[d.wave == pw].set_index("iso3").rel_adults
        for lab, msk in [("EMDE", lambda x: x.emerging == 1), ("DEV", lambda x: x.developing == 1),
                         ("EMDE without Saudi Arabia", lambda x: (x.emerging == 1) & (x.iso3 != "SAU"))]:
            for ibv in ("ib_share", "ib_np"):
                s = p[p.wave.isin((2014, 2017, 2021, 2024)) & msk(p) & (p.ifsb_listed == 1)].copy()
                s["ib"] = s[ibv] / 10
                s["pool"] = s.iso3.map(pool)
                s = s.dropna(subset=["ib", "pool", "fiaccount"])
                s = s[s.groupby("iso3").iso3.transform("size") >= 2]
                s["z"] = s.ib * s.pool
                for y in ("fiaccount", "account", "save_formal"):
                    s["yy"] = 100 * s[y]
                    f = FWLFit(s, "yy", "z", fe=("iso3", "wave"), controls=("ib",), cluster="iso3", scale=1.0)
                    o = f.summary(name="dose: IB/10 x pool", cr2=True)
                    o.update(sample=lab, outcome=y, pool_wave=pw, ibvar=ibv, mean_pool=s.drop_duplicates("iso3").pool.mean(),
                             sd_pool=s.drop_duplicates("iso3").pool.std())
                    rows.append(o)
    return pd.DataFrame(rows)


def main():
    d = measures(pd.read_csv(os.path.join(DATA, "barriers_panel.csv")))
    os.makedirs(RES, exist_ok=True)
    rows = []
    specs = [("EMDE", "ib_share", MAIN_WAVES, "emerging", None),
             ("DEV", "ib_share", MAIN_WAVES, "developing", None),
             ("EMDE, Muslim-majority", "ib_share", MAIN_WAVES, "emerging", lambda s: s[s.muslim2020 >= 50]),
             ("EMDE without Saudi Arabia", "ib_share", MAIN_WAVES, "emerging", lambda s: s[s.iso3 != "SAU"]),
             ("EMDE without Iran", "ib_share", MAIN_WAVES, "emerging", lambda s: s[s.iso3 != "IRN"]),
             ("DEV without Malaysia", "ib_share", MAIN_WAVES, "developing", lambda s: s[s.iso3 != "MYS"]),
             ("ZERO", "ib_z", MAIN_WAVES, "emerging", None),
             ("ZERO without Saudi Arabia and Malaysia", "ib_z", MAIN_WAVES, "emerging",
              lambda s: s[~s.iso3.isin(["SAU", "MYS"])]),
             ("EMDE, interpolated share", "ib_interp", MAIN_WAVES, "emerging", None),
             ("EMDE, unmatched steps neutralised", "ib_np", MAIN_WAVES, "emerging", None),
             ("EMDE, all large steps neutralised", "ib_ns", MAIN_WAVES, "emerging", None),
             ("EMDE, repeated values interpolated", "ib_sx", MAIN_WAVES, "emerging", None),
             ("EMDE, with 2011", "ib_share", (2011,) + MAIN_WAVES, "emerging", None),
             ("EMDE, balanced", "ib_share", MAIN_WAVES, "emerging",
              lambda s: s[s.groupby("iso3").iso3.transform("size") == 3])]
    keep = []
    for lab, ibv, waves, flag, extra in specs:
        s = sample(d, ibv, waves, flag, extra)
        keep.append(s.assign(sample=lab))
        r = fit(s, "L_rel", "log contrast", lab, bound=BOUND_LOG)
        r.update(long_diff(s))
        rows.append(r)
        if lab in ("EMDE", "DEV", "EMDE without Saudi Arabia", "EMDE, unmatched steps neutralised"):
            rows.append(fit(s, "R_rel", "religion share of citations (pp)", lab))
            rows.append(poisson(s, lab))
            rows.append(fit_wls(s, lab))
        if lab == "EMDE":
            for x in OTHERS:
                rows.append(fit(s, f"L_{x}", f"placebo barrier: {x}", lab))
            e = s.copy()
            e["ib_x_m"] = e.ib * e.muslim2020 / 100
            f = FWLFit(e, "L_rel", "ib_x_m", fe=("iso3", "wave"), controls=("ib",), scale=10.0)
            o = f.summary(name="log contrast: Islamic share x Muslim share (100 % minus 0 %)")
            o.update(sample=lab, outcome="L_rel")
            rows.append(o)
    rows += lead_test(d)
    rows += trend_variants(d)
    rows += mode_variants(d)
    res = pd.DataFrame(rows)
    plac = res.name.str.startswith("placebo")
    res.loc[plac, "p_wcr_holm"] = holm(res.loc[plac, "p_wcr"])
    res.to_csv(os.path.join(RES, "barrier_panel_estimates.csv"), index=False)
    pd.concat(keep).to_csv(os.path.join(RES, "barrier_panel_data.csv"), index=False)
    pd.set_option("display.width", 250)
    cols = ["sample", "name", "beta", "p", "p_wcr", "wb_lo", "wb_hi", "p_cr2", "p_cr3", "p_equiv", "G", "n", "g_star",
            "lev_top", "lev_max", "mde80_wcr", "beta_ld", "p_ri_ld"]
    print(res[[c for c in cols if c in res]].round(3).to_string())

    krows = []
    for k in (0.25, 0.5, 1.0, 2.0, 5.0):
        dk = measures(pd.read_csv(os.path.join(DATA, "barriers_panel.csv")), k)
        for lab, ibv, flag, extra in [("EMDE", "ib_share", "emerging", None),
                                      ("EMDE without Saudi Arabia", "ib_share", "emerging", lambda s: s[s.iso3 != "SAU"]),
                                      ("DEV", "ib_share", "developing", None)]:
            r = fit(sample(dk, ibv, MAIN_WAVES, flag, extra), "L_rel", f"log contrast, k = {k:g}", lab)
            r["k"] = k
            krows.append(r)
    kk = pd.DataFrame(krows)
    kk["pct"] = 100 * (np.exp(kk.beta) - 1)
    kk.to_csv(os.path.join(RES, "barrier_panel_k.csv"), index=False)
    print(kk[["k", "sample", "beta", "pct", "p_wcr", "wb_lo", "wb_hi"]].round(3).to_string())

    bw = bound_waves(d)
    bw.to_csv(os.path.join(RES, "bound_waves.csv"), index=False)
    mg = muslim_gap(d)
    mg.to_csv(os.path.join(RES, "muslim_gap.csv"), index=False)
    zt = zero_trend(d)
    zt.to_csv(os.path.join(RES, "zero_trend.csv"), index=False)
    print(zt.round(3).to_string())
    print(mg.round(3).to_string())
    print(bw.round(3).to_string())
    ig = implied_gain(res, d)
    ig.to_csv(os.path.join(RES, "implied_gain.csv"), index=False)
    print(ig.round(3).to_string())
    do = dose(d)
    do.to_csv(os.path.join(RES, "dose_estimates.csv"), index=False)
    print(do[["sample", "outcome", "pool_wave", "ibvar", "beta", "p", "p_wcr", "wb_lo", "wb_hi", "p_cr2", "G", "n",
              "g_star", "lev_top", "lev_max", "mean_pool"]].round(3).to_string())


if __name__ == "__main__":
    main()
