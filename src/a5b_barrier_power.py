"""Design-based power of the barrier-panel test (Proposition 3 alternative) on the observed EMDE design.

The observed EMDE barrier panel, as published and with the unmatched steps neutralised, (countries, waves 2014, 2017, 2021, Islamic shares, numbers of respondents asked)
is kept. Respondent counts are simulated as binomial draws, so sampling noise has its observed size:
  p_ct,b = clip(pbar_cb * exp(d_tb + nu_ctb), 0, 0.99) for the six other reasons,
  p_ct,rel = clip(pbar_c,rel * exp(d_t,rel + nu_ct,rel) * (1 - theta)^((IB_ct - IB_c,first)/10), 0, 0.99),
where pbar_cb is the country mean share of reason b, d_tb the wave mean deviation, and nu_ctb country-wave
noise beyond sampling noise with a standard deviation chosen so that the simulated residual variance of the log
contrast under theta = 0 matches the observed residual variance of the two-way FE regression. The log contrast
(k = 0.5) is regressed on the share with country and wave effects and tested with the WCR bootstrap (999 Webb
draws) and CR1 at 5%. theta from 0 to 0.9, 2,000 replications, seeds 0 and 1. For the published design the simulation is run both for
faith-excluded adults who cite no other reason (q = 0) and for adults who cite every other reason (q = 1), whose
inclusion lowers the other reasons as well. results/barrier_power_summary.csv: reach detected with 80% power.

Outputs: results/barrier_power.csv (power by theta and test), results/barrier_noise.csv (variance decomposition)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FWLFit  # noqa: E402
from a2b_barrier_panel import BARS, OTHERS, K, measures, sample  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
R = 2000
THETAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def contrast(S, k=K):
    la = np.log(S + k)
    ir = BARS.index("religion")
    oth = [i for i in range(len(BARS)) if i != ir]
    return la[:, ir] - la[:, oth].mean(axis=1)


def resid_var(df, y):
    """Residual variance of the two-way FE regression with fixed-effect degrees of freedom."""
    f = FWLFit(df, y, "ib", fe=("iso3", "wave"), cluster="iso3", scale=10.0)
    dof = f.n - (f.G + df.wave.nunique())
    return float(f.resid @ f.resid / dof), f


def run(d, design, ibvar, seed, qs=(0,)):
    s = sample(d, ibvar).reset_index(drop=True)
    n = s.n_asked.values.astype(int)
    P = s[[f"s_{b}" for b in BARS]].values / 100
    pbar = s.groupby("iso3")[[f"s_{b}" for b in BARS]].transform("mean").values / 100
    lp = np.log(np.clip(P, 1e-3, None)) - np.log(np.clip(pbar, 1e-3, None))
    dt = pd.DataFrame(lp).groupby(s.wave.values).transform("mean").values
    ib = s.ib.values
    ib_first = s.groupby("iso3").ib.transform("first").values
    obs_var, fobs = resid_var(s, "L_rel")

    rng = np.random.default_rng(seed)

    def simulate(theta, sd_nu, rng, q=0):
        nu = rng.normal(0, sd_nu, size=P.shape)
        p = pbar * np.exp(dt + nu)
        ir = BARS.index("religion")
        f = 1 - (1 - theta) ** ((ib - ib_first) / 10)       # fraction of the faith-excluded brought in
        moved = f * p[:, ir]
        p[:, ir] = p[:, ir] - moved
        if q == 1:                                           # they also cited every other barrier
            for j in range(len(BARS)):
                if j != ir:
                    p[:, j] = p[:, j] - moved
        p = np.clip(p, 0, 0.99)
        y = rng.binomial(n[:, None], p)
        return 100 * y / n[:, None]

    # calibrate sd_nu: residual variance of the simulated contrast under theta = 0
    def sim_var(sd_nu, reps=200):
        r2 = np.random.default_rng(1)
        v = []
        for _ in range(reps):
            ss = s.copy()
            ss["Lsim"] = contrast(simulate(0.0, sd_nu, r2))
            v.append(resid_var(ss, "Lsim")[0])
        return float(np.mean(v))
    v0 = sim_var(0.0)
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    vs = [sim_var(g) for g in grid]
    if obs_var <= vs[0]:
        sd_nu = 0.0
    else:
        sd_nu = float(np.interp(obs_var, vs, grid))
    noise = pd.DataFrame([dict(design=design, obs_resid_var=obs_var, sampling_only_var=v0, sampling_share=v0 / obs_var, sd_nu=sd_nu,
                               G=fobs.G, n=fobs.n, median_asked=float(np.median(n)))])
    print(noise.round(3).to_string())

    rows = []
    for q, theta in [(q, t) for q in qs for t in THETAS]:
        rej_w = rej_c = 0
        bet = []
        for r in range(R):
            ss = s.copy()
            ss["Lsim"] = contrast(simulate(theta, sd_nu, rng, q))
            f = FWLFit(ss, "Lsim", "ib", fe=("iso3", "wave"), cluster="iso3", scale=10.0)
            wG = f.weights(999, seed=r)
            rej_w += f.wcr_p(0.0, wG) < 0.05
            rej_c += f.p < 0.05
            bet.append(f.beta * 10)
        rows.append(dict(design=design, q=q, theta=theta, true_dL=np.log(1 - theta) if q == 0 else np.nan,
                         mean_beta=np.mean(bet),
                         power_wcr=rej_w / R, power_cr1=rej_c / R, reps=R, sd_nu=sd_nu))
        print(rows[-1], flush=True)
    return noise, pd.DataFrame(rows)


def main():
    d = measures(pd.read_csv(os.path.join(DATA, "barriers_panel.csv")))
    out = [run(d, "EMDE", "ib_share", 0, qs=(0, 1)), run(d, "EMDE, unmatched steps neutralised", "ib_np", 1)]
    pd.concat([o[0] for o in out]).to_csv(os.path.join(RES, "barrier_noise.csv"), index=False)
    pw = pd.concat([o[1] for o in out])
    pw.to_csv(os.path.join(RES, "barrier_power.csv"), index=False)
    # reach detected with 80 % power (linear interpolation on the theta grid)
    summ = []
    for (dsg, q), g in pw.groupby(["design", "q"]):
        g = g.sort_values("theta")
        for test in ("power_wcr", "power_cr1"):
            v = g[test].values
            th = np.nan
            if v.max() >= 0.8:
                i = int(np.argmax(v >= 0.8))
                th = float(np.interp(0.8, v[i - 1:i + 1], g.theta.values[i - 1:i + 1])) if i > 0 else float(g.theta.iloc[0])
            summ.append(dict(design=dsg, q=q, test=test, theta80=th))
    pd.DataFrame(summ).to_csv(os.path.join(RES, "barrier_power_summary.csv"), index=False)
    print(pd.DataFrame(summ))


if __name__ == "__main__":
    main()
