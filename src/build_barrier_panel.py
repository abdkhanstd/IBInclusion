"""Barrier panel for the four Findex waves that asked about religion (2011, 2014, 2017, 2021/22).

Input: data/findex_reasons_codebook.csv (fetch_findex_codebook.py), data/panel.csv (build_panel.py),
data/barriers.csv (published weighted shares 2017 and 2021, build_panel.py).

For each country-wave and each of the seven reasons asked in all four waves, the share of the respondents asked
(adults without an account at a financial institution) who answer yes, yes / (yes + no + don't know + refused),
from the unweighted codebook frequencies. Countries are kept for a wave if all seven reasons are present and at
least 30 respondents were asked.

Outputs
  data/barriers_panel.csv          one row per country-wave: n_asked, yes counts and shares (% of the asked) of
                                   the seven reasons, Islamic share variants, sample flags, FI account ownership,
                                   share of all adults who lack an FI account and cite religion, interview mode
  results/codebook_validation.csv  unweighted codebook shares against the published weighted shares, 2017 and 2021
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
BARS = ["distance", "cost", "documents", "trust", "religion", "funds", "family"]
MIN_ASKED = 30


def codebook():
    c = pd.read_csv(os.path.join(DATA, "findex_reasons_codebook.csv"))
    c = c[c.barrier.isin(BARS)].copy()
    for v in ["yes", "no", "dk", "ref"]:
        c[v] = c[v].fillna(0)
    c["asked"] = c[["yes", "no", "dk", "ref"]].sum(axis=1)
    # a country with two surveys mapped to the same wave (2021 and 2022 releases): keep the first survey year
    c = c.sort_values(["iso3", "wave", "survey_year"])
    first = c.groupby(["iso3", "wave"]).survey_year.transform("min")
    c = c[c.survey_year == first]
    yes = c.pivot_table(index=["iso3", "wave", "survey_year"], columns="barrier", values="yes")
    asked = c.pivot_table(index=["iso3", "wave", "survey_year"], columns="barrier", values="asked")
    ok = yes[BARS].notna().all(axis=1) & (asked[BARS].min(axis=1) >= MIN_ASKED)
    yes, asked = yes[ok], asked[ok]
    out = (100 * yes[BARS] / asked[BARS]).add_prefix("s_")
    out = out.join(yes[BARS].add_prefix("y_"))
    out["n_asked"] = asked[BARS].max(axis=1)
    return out.reset_index()


def main():
    cb = codebook()
    p = pd.read_csv(os.path.join(DATA, "panel.csv"))
    keep = ["iso3", "wave", "year", "country", "ib_share", "ib_interp", "ib_ns", "ib_np", "ib_sx", "ib_z",
            "ifsb_listed", "zero_ok", "emerging", "developing", "region_geo", "muslim", "muslim2020", "fiaccount",
            "account", "lgdppc", "pop_adult"]
    d = cb.merge(p[keep], on=["iso3", "wave"], how="inner")
    # interview mode from the study metadata: face (face-to-face or CAPI), phone, other (not specified)
    m = pd.read_csv(os.path.join(DATA, "findex_survey_mode.csv"))
    ml = m["mode"].fillna("").str.lower()
    m["mode_class"] = np.where(m.phone == 1, "phone",
                               np.where(ml.str.contains("face|capi|papi"), "face", "other"))
    m = m.sort_values(["iso3", "wave", "survey_year"]).drop_duplicates(["iso3", "wave"])
    d = d.merge(m[["iso3", "wave", "mode_class"]], on=["iso3", "wave"], how="left")
    d["mode_class"] = d.mode_class.fillna("other")
    d["fi_unbanked"] = 100 * (1 - d.fiaccount)
    d["rel_adults"] = d.fi_unbanked * d.s_religion / 100   # % of all adults: no FI account and cite religion
    d.to_csv(os.path.join(DATA, "barriers_panel.csv"), index=False)
    print(d.groupby("wave").agg(n=("iso3", "size"), ib=("ib_share", "count"), asked=("n_asked", "median")))

    # validation against the published weighted shares (share of the unbanked, published item base)
    b = pd.read_csv(os.path.join(DATA, "barriers.csv"))
    pub = b.pivot_table(index=["iso3", "wave"], columns="barrier", values="share_unbanked").reset_index()
    pub[BARS] = 100 * pub[BARS]
    v = d.merge(pub, on=["iso3", "wave"])
    rows = []
    for wave in (2017, 2021):
        vv = v[v.wave == wave]
        for x in BARS:
            a, c = vv[f"s_{x}"], vv[x]
            rows.append(dict(wave=wave, barrier=x, n=len(vv), corr=np.corrcoef(a, c)[0, 1],
                             mean_codebook=a.mean(), mean_published=c.mean(), mean_diff=(a - c).mean(),
                             mad=(a - c).abs().mean()))
        # log contrast of religion against the other six (k = 0.5 pp of the unbanked)
        la = np.log(vv[[f"s_{x}" for x in BARS]].values + 0.5)
        lb = np.log(vv[BARS].values + 0.5)
        ia = BARS.index("religion")
        oth = [i for i in range(len(BARS)) if i != ia]
        ca = la[:, ia] - la[:, oth].mean(axis=1)
        cbv = lb[:, ia] - lb[:, oth].mean(axis=1)
        rows.append(dict(wave=wave, barrier="log religion contrast", n=len(vv), corr=np.corrcoef(ca, cbv)[0, 1],
                         mean_codebook=ca.mean(), mean_published=cbv.mean(), mean_diff=(ca - cbv).mean(),
                         mad=np.abs(ca - cbv).mean()))
    # changes 2017 to 2021 of the religion share (countries in both waves)
    w = v.pivot_table(index="iso3", columns="wave", values=["s_religion", "religion"]).dropna()
    da = w[("s_religion", 2021)] - w[("s_religion", 2017)]
    dp = w[("religion", 2021)] - w[("religion", 2017)]
    rows.append(dict(wave="2017-2021 change", barrier="religion", n=len(w), corr=np.corrcoef(da, dp)[0, 1],
                     mean_codebook=da.mean(), mean_published=dp.mean(), mean_diff=(da - dp).mean(),
                     mad=(da - dp).abs().mean()))
    cc = {}
    for wave in (2017, 2021):
        vv = v[v.wave == wave].set_index("iso3")
        la = np.log(vv[[f"s_{x}" for x in BARS]].values + 0.5)
        lb = np.log(vv[BARS].values + 0.5)
        ia = BARS.index("religion")
        oth = [i for i in range(len(BARS)) if i != ia]
        cc[wave] = pd.DataFrame({"cb": la[:, ia] - la[:, oth].mean(axis=1),
                                 "pub": lb[:, ia] - lb[:, oth].mean(axis=1)}, index=vv.index)
    j = cc[2021].join(cc[2017], lsuffix="21", rsuffix="17", how="inner")
    dcb, dpub = j.cb21 - j.cb17, j.pub21 - j.pub17
    rows.append(dict(wave="2017-2021 change", barrier="log religion contrast", n=len(j),
                     corr=np.corrcoef(dcb, dpub)[0, 1], mean_codebook=dcb.mean(), mean_published=dpub.mean(),
                     mean_diff=(dcb - dpub).mean(), mad=(dcb - dpub).abs().mean()))
    emde = d[(d.emerging == 1) & (d.ifsb_listed == 1) & d.ib_share.notna()].iso3.unique()
    je = j[j.index.isin(emde)]
    dcb, dpub = je.cb21 - je.cb17, je.pub21 - je.pub17
    rows.append(dict(wave="2017-2021 change, EMDE", barrier="log religion contrast", n=len(je),
                     corr=np.corrcoef(dcb, dpub)[0, 1], mean_codebook=dcb.mean(), mean_published=dpub.mean(),
                     mean_diff=(dcb - dpub).mean(), mad=(dcb - dpub).abs().mean()))
    out = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    out.to_csv(os.path.join(RES, "codebook_validation.csv"), index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
