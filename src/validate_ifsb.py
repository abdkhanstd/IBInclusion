"""Validate the digitised IFSB shares against values quoted in the text of the same report, and the bar-length
method against the values printed on the bars in the editions that print them (2022, 2023, 2025, 2026).

Each entry: (report edition, jurisdiction as labelled in the chart, value quoted in the text, period of the
quoted value, page-independent quotation key). Only quotations that name a jurisdiction's domestic Islamic
banking share of total banking assets are used. A pair is 'same period' when the quoted period equals the
chart's reference period.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")

QUOTES = [
    # edition 2015 (chart 1H2014)
    (2015, "Yemen", 27.4, "1H2014"), (2015, "Qatar", 25.1, "1H2014"), (2015, "Brunei", 41.0, "1H2014"),
    (2015, "Kuwait", 38.0, "1H2014"), (2015, "Bahrain", 12.7, "1H2014"), (2015, "Jordan", 11.7, "1H2014"),
    # edition 2016 (chart 1H2015)
    (2016, "Yemen", 33.0, "1H2015"), (2016, "Kuwait", 38.9, "1H2015"), (2016, "Qatar", 26.1, "1H2015"),
    (2016, "Malaysia", 23.0, "1H2015"), (2016, "Bangladesh", 19.4, "1H2015"), (2016, "UAE", 18.4, "1H2015"),
    (2016, "Bahrain", 13.5, "1H2015"), (2016, "Oman", 6.5, "1H2015"),
    # edition 2017 (chart 2Q2016)
    (2017, "Saudi Arabia", 51.1, "1H2016"), (2017, "Malaysia", 23.8, "2Q2016"), (2017, "Jordan", 15.2, "1H2016"),
    # edition 2018 (chart 1H2017)
    (2018, "Brunei", 61.8, "1H2017"), (2018, "Saudi Arabia", 51.5, "1H2017"), (2018, "Kuwait", 39.3, "1H2017"),
    (2018, "Bahrain", 14.1, "1H2017"), (2018, "Senegal", 5.0, "1H2017"),
    (2018, "Bosnia & Herzegovina", 3.6, "1H2017"), (2018, "Kyrgyz", 1.5, "1H2017"), (2018, "Tanzania", 0.7, "1H2017"),
    # edition 2019 (chart 2Q2018)
    (2019, "Brunei", 63.6, "2Q2018"), (2019, "Saudi Arabia", 51.5, "2Q2018"), (2019, "Palestine", 14.6, "2Q2018"),
    (2019, "Afghanistan", 9.1, "2Q2018"),
    # edition 2020 (chart 2Q2019; text mostly 3Q2019)
    (2020, "Saudi Arabia", 69.0, "3Q2019"), (2020, "Brunei", 62.8, "3Q2019"), (2020, "Qatar", 26.1, "3Q2019"),
    (2020, "Palestine", 15.9, "3Q2019"),
    # edition 2021 (chart 2Q2020; text 3Q2020)
    (2021, "Saudi Arabia", 68.0, "3Q2020"), (2021, "Brunei", 61.9, "3Q2020"), (2021, "Kuwait", 44.0, "3Q2020"),
    (2021, "Qatar", 26.9, "3Q2020"), (2021, "UAE", 19.0, "3Q2020"), (2021, "Bangladesh", 21.9, "3Q2020"),
    (2021, "Pakistan", 16.0, "3Q2020"), (2021, "Oman", 14.2, "3Q2020"), (2021, "Afghanistan", 12.4, "3Q2020"),
    (2021, "Algeria", 2.4, "3Q2020"), (2021, "Tunisia", 5.1, "3Q2020"), (2021, "Tanzania", 0.8, "3Q2020"),
    (2021, "Kenya", 1.0, "3Q2020"), (2021, "Sri Lanka", 0.87, "3Q2020"), (2021, "Nigeria", 0.7, "3Q2020"),
    (2021, "South Africa", 0.1, "3Q2020"),
    # edition 2022 (chart 2Q2021)
    (2022, "Afghanistan", 13.2, "2021"),
]


def main():
    d = pd.read_csv(os.path.join(DATA, "ifsb_share_digitized.csv"))
    rows = []
    for rep, j, v, per in QUOTES:
        m = d[(d.report == rep) & (d.jurisdiction == j)]
        if not len(m):
            continue
        r = m.iloc[0]
        rows.append(dict(report=rep, jurisdiction=j, quoted=v, quoted_period=per, chart_period=r.ref_period,
                         chart_ref=r.share_ref, chart_tick=r.share_tick,
                         same_period=per == r.ref_period))
    v = pd.DataFrame(rows)
    v["err_ref"] = v.chart_ref - v.quoted
    v["err_tick"] = v.chart_tick - v.quoted
    os.makedirs(RES, exist_ok=True)
    v.to_csv(os.path.join(RES, "ifsb_validation.csv"), index=False)
    summ = []
    for lab, s in [("same period", v[v.same_period]), ("adjacent quarter", v[~v.same_period]), ("all", v)]:
        summ.append(dict(subset=lab, n=len(s), mae_ref=s.err_ref.abs().mean(), median_ae_ref=s.err_ref.abs().median(),
                         max_ae_ref=s.err_ref.abs().max(), corr_ref=np.corrcoef(s.chart_ref, s.quoted)[0, 1],
                         share_within_0p5=(s.err_ref.abs() <= 0.5).mean(),
                         n_tick=int(s.err_tick.notna().sum()), mae_tick=s.err_tick.abs().mean()))
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(RES, "ifsb_validation_summary.csv"), index=False)
    info = pd.read_csv(os.path.join(DATA, "ifsb_digitize_info.csv"))
    # bar length vs printed value, by edition
    pr = d[d.share_printed.notna()].copy()
    pr["err"] = pr.share_bar - pr.share_printed
    pp = pr.groupby(["report", "ref_period"]).agg(n=("err", "size"), mae=("err", lambda e: e.abs().mean()),
                                                  median_ae=("err", lambda e: e.abs().median()),
                                                  max_ae=("err", lambda e: e.abs().max()),
                                                  within_0p5=("err", lambda e: (e.abs() <= 0.5).mean())).reset_index()
    pp.to_csv(os.path.join(RES, "ifsb_validation_printed.csv"), index=False)
    pr[["report", "ref_period", "jurisdiction", "share_bar", "share_printed", "err"]].to_csv(
        os.path.join(RES, "ifsb_validation_printed_rows.csv"), index=False)
    print(pp.round(3).to_string())
    print(summ.round(3).to_string())
    print(v[v.err_ref.abs() > 1].round(2).to_string())
    print(info[["report", "ref_period", "n_bars", "mean_abs_tick_minus_ref", "max_abs_tick_minus_ref"]].round(3))


if __name__ == "__main__":
    main()
