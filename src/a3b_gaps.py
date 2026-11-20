"""Components of the within-country account gaps: account ownership of the richest 60 % and poorest 40 %, and of
adults with secondary or more and primary or less education, two-way FE on the Islamic share (2014-2024).

Output: results/gap_components.csv
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FWLFit  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")


def main():
    p = pd.read_csv(os.path.join(DATA, "panel.csv"))
    b = p[p.wave >= 2014].copy()
    for c in ["account_rich", "account_poor", "account_hiedu", "account_lowedu"]:
        b[c] = 100 * b[c]
    b["gap_edu"] = b.account_hiedu - b.account_lowedu
    S = {"EMDE": b[(b.emerging == 1) & b.ib_share.notna()], "DEV": b[(b.developing == 1) & b.ib_share.notna()]}
    rows = []
    for s, d in S.items():
        d = d.assign(ib=d.ib_share)
        for y in ["account_rich", "account_poor", "account_hiedu", "account_lowedu", "gap_edu"]:
            f = FWLFit(d, y, "ib", fe=["iso3", "wave"], controls=["lgdppc"], scale=10.0)
            r = f.summary(B=99999, jack=True)
            rows.append(dict(sample=s, outcome=y, beta=r["beta"], p_wcr=r["p_wcr"], wb_lo=r["wb_lo"], wb_hi=r["wb_hi"],
                             p_cr3=r["p_cr3"], G=r["G"]))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "gap_components.csv"), index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
