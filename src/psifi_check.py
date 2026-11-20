"""Cross-check of the large steps in the Islamic banking share against the IFSB PSIFI country data previews.

The previews report, for each period, structural indicators of the Islamic banking institutions that a
jurisdiction reports to the IFSB Prudential and Structural Islamic Financial Indicators (PSIFI): ST01, the number
of reporting institutions, and ST03, their total assets (currency and units as printed in the preview: national
currency for Saudi Arabia and Kuwait, USD for Bahrain, millions). ST03 is not the numerator of the share. In Saudi
Arabia the reporting population is four full-fledged Islamic banks, and ST03 is at most about 60 % of the Islamic
banking assets implied by the share: the ratio below uses deposit-money-bank claims on the domestic non-financial
sector (World Bank GFDD.DI.02), which understate total bank assets, so it is an upper bound. Islamic windows of
conventional banks are therefore outside ST03. For Kuwait the same upper bound (1.7 to 2.0) is uninformative, and
for Bahrain no ratio can be computed (GFDD ends before ST03 starts). The check therefore asks only whether a step in
the share occurred in the reporting Islamic institutions; it cannot tell a reclassification from growth of
activity outside them (for example Islamic windows).

Parsing: the period list is read from the header of page 1 (labels split into "2018" and "Q2" are joined), the
ST01 and ST03 rows are read in column order, including the placeholders for missing values, and the number of
cells must equal the number of periods. Files whose rows do not match are skipped and listed.

Classification of a step from period p0 to p1: g_share and g_psifi are the relative changes of the share and of
ST03; d = (1 + g_psifi) / (1 + g_share) - 1 measures how far they diverge. A step is "unmatched" when |d| > 0.10.
The classification of every covered step is the same for any threshold between 0.09 and 0.14.

Outputs: results/psifi_check.csv (ST01 and ST03 by period), results/psifi_steps.csv (every covered step of at
least 5 pp), results/psifi_coverage.csv (upper bound on ST03 relative to the Islamic banking assets implied by the
share, using deposit-money-bank claims on the domestic non-financial sector from the World Bank Global Financial
Development Database, September 2022 version, as a lower bound on total bank assets)
"""
import json
import os
import re
import urllib.request

import numpy as np
import pandas as pd
import pymupdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data_raw", "ifsb", "psifi")
WDI_RAW = os.path.join(ROOT, "data_raw", "wdi")
RES = os.path.join(ROOT, "results")
PER = re.compile(r"^(20\d\d)(A|Q[1-4]|H[12])$")
NUM = re.compile(r"^-?[\d,]+\.?\d*$")
THRESH = 0.10


def lines(page):
    """Words grouped into text lines by vertical position, each line sorted by x."""
    out = {}
    for w in page.get_text("words"):
        out.setdefault(round((w[1] + w[3]) / 2), []).append(w)
    merged = {}
    for y in sorted(out):
        key = next((k for k in merged if abs(k - y) <= 2), y)
        merged.setdefault(key, []).extend(out[y])
    return {k: sorted(v, key=lambda w: w[0]) for k, v in merged.items()}


def periods(page):
    best = []
    for _, ws in lines(page).items():
        toks, i = [], 0
        while i < len(ws):
            t = ws[i][4]
            if re.fullmatch(r"20\d\d", t) and i + 1 < len(ws) and re.fullmatch(r"A|Q[1-4]|H[12]", ws[i + 1][4]):
                t += ws[i + 1][4]
                i += 1
            if PER.match(t):
                toks.append(t)
            i += 1
        if len(toks) > len(best):
            best = toks
    return best


def row_cells(page, code):
    for _, ws in lines(page).items():
        labels = [w for w in ws if w[4] == code]
        if not labels:
            continue
        x0 = labels[0][2]
        cells = [w[4] for w in ws if w[0] > x0]
        # drop the currency and unit columns (for example "NC M", "USD M", "n.a G")
        while cells and not (NUM.match(cells[0]) or cells[0] in ("…", "...")):
            cells = cells[1:]
        unit = " ".join(w[4] for w in ws if x0 < w[0] and w[4] in ("NC", "USD", "M", "B"))
        return cells, unit
    return None, None


def parse(path):
    doc = pymupdf.open(path)
    per = periods(doc[0])
    for page in doc:
        st01, _ = row_cells(page, "ST01")
        st03, unit = row_cells(page, "ST03")
        if st03 is None:
            continue
        if len(st03) != len(per) or (st01 is not None and len(st01) != len(per)):
            return per, None, f"cells {len(st03)} / {None if st01 is None else len(st01)} vs periods {len(per)}"

        def num(x):
            return float(x.replace(",", "")) if NUM.match(x) else np.nan
        return per, pd.DataFrame(dict(period=per, st01=[num(x) for x in st01] if st01 else np.nan,
                                      st03=[num(x) for x in st03], unit=unit)), "ok"
    return per, None, "no ST03 row"


def qtime(p):
    y = int(p[:4])
    return y + {"A": 0.9, "Q1": 0.1, "Q2": 0.4, "Q3": 0.6, "Q4": 0.9, "H1": 0.4, "H2": 0.9}[p[4:]]


def gfdd(isos):
    """GFDD.DI.02 (deposit money banks' claims on the domestic non-financial sector, % of GDP; World Bank Global
    Financial Development Database) and GDP in current local currency, World Bank API (cached)."""
    f = os.path.join(WDI_RAW, "gfdd_bank_assets.json")
    if not os.path.exists(f):
        out = {}
        for ind in ["GFDD.DI.02", "NY.GDP.MKTP.CN"]:
            url = (f"https://api.worldbank.org/v2/country/{';'.join(isos)}/indicator/{ind}?format=json&date=2013:2023"
                   f"&per_page=500")
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ibinclusion/1.0"}),
                                        timeout=120) as r:
                out[ind] = json.load(r)[1]
        os.makedirs(WDI_RAW, exist_ok=True)
        json.dump(out, open(f, "w"))
    raw = json.load(open(f))
    rows = []
    for ind, recs in raw.items():
        for r in recs or []:
            if r["value"] is not None:
                rows.append(dict(iso3=r["countryiso3code"], year=int(r["date"]), ind=ind, value=r["value"]))
    d = pd.DataFrame(rows).pivot_table(index=["iso3", "year"], columns="ind", values="value").reset_index()
    return d


def main():
    rows, status = [], []
    for c, iso in [("Saudi-Arabia", "SAU"), ("Kuwait", "KWT"), ("Bahrain", "BHR"), ("Malaysia", "MYS"),
                   ("Pakistan", "PAK"), ("Bangladesh", "BGD"), ("Egypt", "EGY"), ("Jordan", "JOR")]:
        f = os.path.join(RAW, c + ".pdf")
        if not os.path.exists(f):
            continue
        per, d, msg = parse(f)
        status.append(dict(iso3=iso, n_periods=len(per), first=per[0] if per else "", last=per[-1] if per else "",
                           status=msg, n_st03=int(d.st03.notna().sum()) if d is not None else 0))
        if d is not None:
            d["iso3"] = iso
            d["t"] = d.period.map(qtime)
            rows.append(d)
    st = pd.DataFrame(status)
    print(st.to_string())
    d = pd.concat(rows)
    os.makedirs(RES, exist_ok=True)
    d[["iso3", "period", "t", "st01", "st03", "unit"]].to_csv(os.path.join(RES, "psifi_check.csv"), index=False)

    a = pd.read_csv(os.path.join(ROOT, "data", "ifsb_audit.csv")).sort_values(["iso3", "t"])
    a = a[(a.carried_forward == 0) & (a.outlier == 0)].copy()
    a["prev_period"] = a.groupby("iso3").ref_period.shift(1)
    a["prev_t"] = a.groupby("iso3").t.shift(1)
    a["prev_share"] = a.groupby("iso3").share_pct.shift(1)
    out = []
    for _, r in a[a.step == 1].iterrows():
        g = d[(d.iso3 == r.iso3) & d.st03.notna()]
        if g.empty or r.prev_t < g.t.min() or r.t > g.t.max():
            continue
        v0, v1 = np.interp(r.prev_t, g.t, g.st03), np.interp(r.t, g.t, g.st03)
        n0, n1 = np.interp(r.prev_t, g.t, g.st01), np.interp(r.t, g.t, g.st01)
        out.append(dict(iso3=r.iso3, from_period=r.prev_period, to_period=r.ref_period, share_from=r.prev_share,
                        share_to=r.share_pct, st01_from=n0, st01_to=n1, psifi_assets_from=v0, psifi_assets_to=v1,
                        unit=g.unit.iloc[0], psifi_growth_pct=100 * (v1 / v0 - 1)))
    o = pd.DataFrame(out)
    o["share_growth_pct"] = 100 * (o.share_to / o.share_from - 1)
    o["divergence_pct"] = 100 * ((1 + o.psifi_growth_pct / 100) / (1 + o.share_growth_pct / 100) - 1)
    o["unmatched"] = (o.divergence_pct.abs() > 100 * THRESH).astype(int)
    o.to_csv(os.path.join(RES, "psifi_steps.csv"), index=False)
    print(o.round(1).to_string())
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_panel import UNMATCHED_STEPS
    found = set(zip(o.iso3[o.unmatched == 1], o["to_period"][o.unmatched == 1]))
    assert found == UNMATCHED_STEPS, (found, UNMATCHED_STEPS)

    # coverage of ST03 relative to the Islamic assets implied by the share (annual, where GFDD reports bank assets)
    try:
        gf = gfdd(["SAU", "KWT", "BHR"])
    except Exception as e:  # offline: skip the coverage table
        print("GFDD not available:", e)
        return
    cov = []
    for _, r in gf.dropna().iterrows():
        g = d[(d.iso3 == r.iso3) & d.st03.notna()]
        s = a[a.iso3 == r.iso3]
        tt = r.year + 0.9
        if g.empty or s.empty or not (g.t.min() <= tt <= g.t.max()) or not (s.t.min() <= tt <= s.t.max()):
            continue
        total = r["GFDD.DI.02"] / 100 * r["NY.GDP.MKTP.CN"] / 1e6       # domestic claims, local currency, millions
        share = np.interp(tt, s.t, s.share_pct) / 100
        st03 = np.interp(tt, g.t, g.st03)
        cov.append(dict(iso3=r.iso3, year=int(r.year), dmb_domestic_claims_lc_m=total, share_pct=100 * share,
                        st03=st03, unit=g.unit.iloc[0], st03_over_implied_islamic_upper=st03 / (share * total)))
    cv = pd.DataFrame(cov)
    cv.to_csv(os.path.join(RES, "psifi_coverage.csv"), index=False)
    print(cv.round(3).to_string())


if __name__ == "__main__":
    main()
