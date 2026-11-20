"""Q4: inclusion or migration? Annual country panel 2013-2024 (IMF Financial Access Survey via WDI).

If a rise in the Islamic share of banking assets brings new customers into the banking system, depositors and
borrowers per 1,000 adults rise with it. If it reflects customers and assets moving from conventional to
Islamic banks (or growth of corporate and sovereign business, or reclassification), these counts do not move.
With conventional assets fixed, a 10 pp rise in the Islamic share raises total banking assets by at least
10.5 % (log 0.105) if all of it is new business. If a fraction phi of the growth comes from new customers with
average balances, log customer counts rise by about phi * 0.10 per 10 pp. We test phi = 0.1, 0.25, 0.5, 1.

All bootstrap results use 99,999 draws. The phi tests are one-sided and not adjusted for multiplicity.
The share is also used with the unmatched steps, or all steps of at least 5 pp, neutralised (build_panel.py).
The event study is reported with all six documented conversions and without Saudi Arabia, whose event year is
the year of the step in its share.

Outputs: results/annual_estimates.csv, annual_leadlag.csv, annual_events.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FWLFit  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
OUTCOMES = ["ldepositors", "lborrowers", "branches", "privcredit", "lfp_female", "lfp_male", "lfp_gap", "selfemp"]
PHI = [0.1, 0.25, 0.5, 1.0]
EVENTS = {"PAK": 2015, "IDN": 2016, "KAZ": 2017, "AFG": 2018, "SAU": 2019, "BGD": 2020}


def main():
    a = pd.read_csv(os.path.join(DATA, "annual.csv"))
    a["lfp_gap"] = a.lfp_female - a.lfp_male
    S = {
        "EMDE": a[(a.emerging == 1) & (a.ifsb_listed == 1)],
        "DEV": a[(a.developing == 1) & (a.ifsb_listed == 1)],
        "EMDE without Saudi Arabia": a[(a.emerging == 1) & (a.ifsb_listed == 1) & (a.iso3 != "SAU")],
        "EXT": a[a.ifsb_listed == 1],
        "ZERO": a[a.emerging == 1],
        "EMDE, unmatched steps neutralised": a[(a.emerging == 1) & (a.ifsb_listed == 1)].assign(ib=lambda d: d.ib_np),
        "EMDE, all large steps neutralised": a[(a.emerging == 1) & (a.ifsb_listed == 1)].assign(ib=lambda d: d.ib_ns),
        "EMDE, repeated values interpolated": a[(a.emerging == 1) & (a.ifsb_listed == 1)].assign(ib=lambda d: d.ib_sx),
    }
    rows = []
    for sname, d in S.items():
        full = sname in ("EMDE", "DEV")
        specs = [("TWFE", ["iso3", "year"], ["lgdppc"])]
        if full:
            specs += [("TWFE+region-year", ["iso3", ("region", "year")], ["lgdppc"]),
                      ("TWFE, no GDP control", ["iso3", "year"], [])]
        for y in OUTCOMES:
            for est, fe, ctr in specs:
                dd = d.dropna(subset=[y, "ib"] + ctr)
                if dd.iso3.nunique() < 8:
                    continue
                f = FWLFit(dd, y, "ib", fe=fe, controls=ctr, scale=10.0)
                r = f.summary(name=y, B=99999, jack=full, ci90=full, cr2=full and est == "TWFE")
                if y in ("ldepositors", "lborrowers"):
                    wG = f.weights(99999, 0)
                    for phi in PHI:
                        r[f"p_phi{phi:g}"] = f.wcr_p(phi * 0.10 / 10, wG, side="less")
                r.update(sample=sname, estimator=est, outcome=y)
                rows.append(r)
    res = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    res.to_csv(os.path.join(RES, "annual_estimates.csv"), index=False)

    # leads and lags with the bootstrap: each coefficient estimated with the other two as controls
    ll = []
    for sname in ["EMDE", "DEV"]:
        d = S[sname].sort_values(["iso3", "year"]).copy()
        g = d.groupby("iso3").ib
        d["ib_lead2"] = g.shift(-2)
        d["ib_lag2"] = g.shift(2)
        for y in ["ldepositors", "lborrowers", "branches", "privcredit", "lfp_gap"]:
            dd = d.dropna(subset=[y, "ib", "ib_lead2", "ib_lag2", "lgdppc"])
            if dd.iso3.nunique() < 8:
                continue
            terms = ["ib_lead2", "ib", "ib_lag2"]
            for k in terms:
                ctr = [t for t in terms if t != k] + ["lgdppc"]
                f = FWLFit(dd, y, k, fe=["iso3", "year"], controls=ctr, scale=10.0)
                r = f.summary(B=99999, jack=False)
                ll.append(dict(sample=sname, outcome=y, term=k, beta=r["beta"], se=r["se"], p=r["p"],
                               p_wcr=r["p_wcr"], wb_lo=r["wb_lo"], wb_hi=r["wb_hi"], wb_set=r["wb_set"], n=r["n"],
                               G=r["G"]))
    pd.DataFrame(ll).to_csv(os.path.join(RES, "annual_leadlag.csv"), index=False)

    # event study around documented conversions (EMDE, IFSB-listed): post-event indicator, two-year pre-event
    # indicator as control; country and year FE, log GDP per capita
    ev = []
    for variant, d0 in [("all six events", S["EMDE"]), ("without Saudi Arabia", S["EMDE without Saudi Arabia"])]:
        d = d0.copy()
        d["ev"] = d.iso3.map(EVENTS)
        d["post"] = ((d.year >= d.ev) & d.ev.notna()).astype(float)
        d["pre2"] = ((d.year >= d.ev - 2) & (d.year < d.ev) & d.ev.notna()).astype(float)
        for y in ["ib", "ldepositors", "lborrowers", "branches", "privcredit"]:
            dd = d.dropna(subset=[y, "lgdppc"])
            for k, other in [("post", "pre2"), ("pre2", "post")]:
                f = FWLFit(dd, y, k, fe=["iso3", "year"], controls=[other, "lgdppc"])
                r = f.summary(B=99999, jack=True)
                ev.append(dict(variant=variant, outcome=y, term=k, beta=r["beta"], se=r["se"], p=r["p"],
                               p_wcr=r["p_wcr"], wb_lo=r["wb_lo"], wb_hi=r["wb_hi"], wb_set=r["wb_set"],
                               p_cr3=r["p_cr3"], n=r["n"], G=r["G"], treated=int(dd[dd.ev.notna()].iso3.nunique())))
    pd.DataFrame(ev).to_csv(os.path.join(RES, "annual_events.csv"), index=False)
    pd.set_option("display.width", 260)
    cols = ["sample", "estimator", "outcome", "beta", "p_wcr", "wb_lo", "wb_hi", "p_phi0.1", "p_phi0.25", "p_phi0.5",
            "p_phi1", "p_cr3", "lev_max", "lev_top", "G"]
    print(res[[c for c in cols if c in res]].round(3).to_string())
    print(pd.DataFrame(ll).round(3).to_string())
    print(pd.DataFrame(ev).round(3).to_string())


if __name__ == "__main__":
    main()
