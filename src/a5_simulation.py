"""Q5: design-based Monte Carlo on the observed EMDE design (countries, waves 2014-2024 and Islamic shares).

  Y_T = a_c + d_t + beta   * IB_ct + lam_T * S_ct + e_T          (target outcome)
  Y_P = a_c + d'_t + beta_P * IB_ct + lam_P * S_ct + e_P          (matched placebo outcome)
  a_c  = sd_a * (kappa * z(mean_t IB_ct) + sqrt(1 - kappa^2) * N(0,1))        cross-country confounding
  S_ct = sd_s * (rho * z(IB_ct - mean_t IB_ct) + sqrt(1 - rho^2) * N(0,1))    shock confounding
  e    ~ N(0, sigma_c^2) with country-specific sigma_c taken from the observed two-way FE residuals of formal
         saving (heteroskedastic, leverage-concentrated errors), rescaled to a mean of 3 pp.
By construction D is unbiased when lam_P = lam_T and beta_P = 0, and its bias follows the identity
bias_D = (1 - lam_P / lam_T) * bias_B2 - beta_P. The simulation is therefore used for inference and power, not
to establish bias: it compares CR1 with the wild cluster restricted bootstrap (WCR, 399 Webb draws per
replication) when the assumptions fail (lam_P/lam_T in {0.5, -0.5}; an absolute placebo effect beta_P of
-0.1 or +0.1, that is -1 or +1 pp per 10 pp, so that the size rows are informative).
Power by outcome: the B2 and D rejection rates for a true effect of 2 pp per 10 pp when the country-specific error
SDs are those of each target outcome's own two-way FE residuals (not rescaled).

Output: results/simulation.csv, results/simulation_by_seed.csv, results/simulation_power_by_outcome.csv
"""
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
R = 1000
SEEDS = (0, 1, 2, 3, 4)
BW = 399
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def dummies(codes):
    k = codes.max() + 1
    D = np.zeros((len(codes), k))
    D[np.arange(len(codes)), codes] = 1
    return D[:, 1:]


class Est:
    def __init__(self, Z, x, groups):
        P = Z @ np.linalg.pinv(Z)
        self.M = np.eye(len(x)) - P
        self.xt = self.M @ x
        self.sxx = self.xt @ self.xt
        self.G = groups.max() + 1
        n, k = len(x), np.linalg.matrix_rank(Z) + 1
        self.adj = (self.G / (self.G - 1)) * ((n - 1) / (n - k))
        self.H = np.zeros((self.G, n))
        self.H[groups, np.arange(n)] = 1
        self.C = self.H @ (self.xt ** 2)

    def run(self, Y, wG):
        Yt = self.M @ Y
        b = self.xt @ Yt / self.sxx
        U = Yt - np.outer(self.xt, b)
        S = self.H @ (self.xt[:, None] * U)
        se = np.sqrt(self.adj * np.sum(S ** 2, axis=0)) / self.sxx
        # WCR p-value for beta = 0: restricted residuals are Yt, cluster sums A_g = sum x~ y~
        A = self.H @ (self.xt[:, None] * Yt)                  # G x R
        a = wG @ A / self.sxx                                 # B x R
        c = (wG @ self.C) / self.sxx                          # B
        d = a                                                 # beta* - 0
        Sst = wG[:, :, None] * A[None, :, :] - d[:, None, :] * self.C[None, :, None]   # B x G x R
        se_st = np.sqrt(self.adj * np.sum(Sst ** 2, axis=1)) / self.sxx               # B x R
        tst = d / se_st
        t0 = b / se
        p_wcr = np.mean(np.abs(tst) >= np.abs(t0)[None, :], axis=0)
        return b, se, p_wcr


def main():
    p = pd.read_csv(os.path.join(DATA, "panel.csv"))
    d = p[(p.emerging == 1) & p.ib_share.notna() & (p.wave >= 2014)].reset_index(drop=True)
    ib = d.ib_share.values.astype(float)
    c = pd.factorize(d.iso3)[0]
    t = pd.factorize(d.wave)[0]
    G, n, T = c.max() + 1, len(d), t.max() + 1
    one = np.ones((n, 1))
    E = {"B1": Est(np.hstack([one, dummies(t)]), ib, c), "B2": Est(np.hstack([one, dummies(c), dummies(t)]), ib, c)}
    E["D"] = E["B2"]
    # country-specific residual SDs from the observed formal-saving two-way FE residuals
    y = 100 * d.save_formal.values
    ok = ~np.isnan(y)
    Mfull = E["B2"].M
    res = np.full(n, np.nan)
    sub = np.where(ok)[0]
    Zs = np.hstack([one, dummies(c), dummies(t)])[sub]
    coef = np.linalg.lstsq(np.column_stack([Zs, ib[sub]]), y[sub], rcond=None)[0]
    res[sub] = y[sub] - np.column_stack([Zs, ib[sub]]) @ coef
    sd_c = pd.Series(res).groupby(c).std().reindex(range(G)).fillna(pd.Series(res).std()).values
    sd_c = np.clip(sd_c, 0.5, None)
    sd_c = 3.0 * sd_c / sd_c.mean()
    qcrit = stats.t.ppf(0.975, G - 1)
    mean_c = pd.Series(ib).groupby(c).transform("mean").values
    cm = pd.Series(ib).groupby(c).mean()
    zbar = ((cm - ib.mean()) / cm.std()).values[c]
    within = ib - mean_c
    zw = within / within.std()
    sd_a, sd_d, sd_s = 10.0, 3.0, 3.0
    scen = [("No confounding", 0.0, 0.0, 1.0, 0.0), ("Cross-country, kappa 0.5", 0.5, 0.0, 1.0, 0.0),
            ("Shock, rho 0.3", 0.0, 0.3, 1.0, 0.0), ("Shock, rho 0.6", 0.0, 0.6, 1.0, 0.0),
            ("Shock rho 0.6, half loading", 0.0, 0.6, 0.5, 0.0), ("Shock rho 0.6, negative loading", 0.0, 0.6, -0.5, 0.0),
            ("Placebo effect -1 pp per 10 pp", 0.0, 0.0, 1.0, -0.1),
            ("Placebo effect +1 pp per 10 pp", 0.0, 0.0, 1.0, 0.1)]
    rows = []
    t0 = time.time()
    for lab, kappa, rho, lam, bp in scen:
        for beta in [0.0, 0.2]:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                wG = WEBB[rng.integers(0, 6, size=(BW, G))]
                a = sd_a * (kappa * zbar[:, None] + np.sqrt(1 - kappa ** 2) * rng.standard_normal((G, R))[c])
                dT = sd_d * rng.standard_normal((T, R))[t]
                dP = sd_d * rng.standard_normal((T, R))[t]
                S = sd_s * (rho * zw[:, None] + np.sqrt(1 - rho ** 2) * rng.standard_normal((n, R)))
                eT = sd_c[c][:, None] * rng.standard_normal((n, R))
                eP = sd_c[c][:, None] * rng.standard_normal((n, R))
                YT = a + dT + beta * ib[:, None] + S + eT
                YP = a + dP + bp * ib[:, None] + lam * S + eP
                for name, Y in [("B1", YT), ("B2", YT), ("D", YT - YP)]:
                    b, se, pw = E[name].run(Y, wG)
                    lo, hi = b - qcrit * se, b + qcrit * se
                    rows.append(dict(scenario=lab, kappa=kappa, rho=rho, lam_ratio=lam, placebo_effect=bp, beta=beta,
                                     seed=seed, estimator=name, bias=float(np.mean(b - beta)),
                                     rmse=float(np.sqrt(np.mean((b - beta) ** 2))),
                                     coverage=float(np.mean((lo <= beta) & (beta <= hi))),
                                     reject_cr1=float(np.mean(np.abs(b / se) > qcrit)),
                                     reject_wcr=float(np.mean(pw < 0.05)), mean_se=float(np.mean(se))))
    res = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    res.to_csv(os.path.join(RES, "simulation_by_seed.csv"), index=False)
    agg = res.groupby(["scenario", "kappa", "rho", "lam_ratio", "placebo_effect", "beta", "estimator"], sort=False).agg(
        bias=("bias", "mean"), bias_sd=("bias", "std"), rmse=("rmse", "mean"), coverage=("coverage", "mean"),
        reject_cr1=("reject_cr1", "mean"), reject_wcr=("reject_wcr", "mean"), reject_wcr_sd=("reject_wcr", "std"),
        mean_se=("mean_se", "mean")).reset_index()
    agg.to_csv(os.path.join(RES, "simulation.csv"), index=False)
    pd.set_option("display.width", 250)
    print(agg.round(3).to_string())
    print("runtime_s", round(time.time() - t0, 1), "n", n, "G", G)

    # power at the observed error scale of each target outcome (and of its placebo for D)
    pw_rows = []
    pairs = [("save_formal", "save_club"), ("borrow_formal", "borrow_family"), ("borrow_business", "borrow_health"),
             ("lfp_female", "lfp_male")]

    def sds(var):
        yv = d[var].values.astype(float) * (100 if var not in ("lfp_female", "lfp_male") else 1)
        okv = ~np.isnan(yv)
        sb = np.where(okv)[0]
        Zv = np.hstack([one, dummies(c), dummies(t)])[sb]
        cf = np.linalg.lstsq(np.column_stack([Zv, ib[sb]]), yv[sb], rcond=None)[0]
        rr = np.full(n, np.nan)
        rr[sb] = yv[sb] - np.column_stack([Zv, ib[sb]]) @ cf
        out = pd.Series(rr).groupby(c).std().reindex(range(G))
        return np.clip(out.fillna(out.mean()).values, 0.25, None)

    for tv, pv in pairs:
        sT, sP = sds(tv), sds(pv)
        for seed in SEEDS:
            rng = np.random.default_rng(100 + seed)
            wG = WEBB[rng.integers(0, 6, size=(BW, G))]
            eT = sT[c][:, None] * rng.standard_normal((n, R))
            eP = sP[c][:, None] * rng.standard_normal((n, R))
            YT = 0.2 * ib[:, None] + eT
            for name, Y in [("B2", YT), ("D", YT - eP)]:
                b, se, pwv = E[name].run(Y, wG)
                pw_rows.append(dict(target=tv, placebo=pv, estimator=name, seed=seed, mean_sd_target=float(sT.mean()),
                                    mean_sd_placebo=float(sP.mean()), reject_wcr=float(np.mean(pwv < 0.05)),
                                    reject_cr1=float(np.mean(np.abs(b / se) > qcrit))))
    pw = pd.DataFrame(pw_rows).groupby(["target", "placebo", "estimator"], sort=False).agg(
        mean_sd_target=("mean_sd_target", "first"), mean_sd_placebo=("mean_sd_placebo", "first"),
        power_wcr=("reject_wcr", "mean"), power_cr1=("reject_cr1", "mean")).reset_index()
    pw.to_csv(os.path.join(RES, "simulation_power_by_outcome.csv"), index=False)
    print(pw.round(3).to_string())


if __name__ == "__main__":
    main()
