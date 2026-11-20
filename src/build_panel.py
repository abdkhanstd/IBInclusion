"""Build the country-wave analysis panel.

Inputs (code/data_raw):
  findex/GlobalFindexDatabase2025.csv     usage outcomes, waves 2011-2024 (World Bank, 2025 release)
  findex/Findex2017_Database.xlsx         barriers to account ownership, wave 2017 (2018 release)
  findex/Findex2021_DatabankWide.xlsx     barriers to account ownership, wave 2021/22 (2022 release)
  pew/pew_religious_composition_2010_2020.csv   Muslim population share, 2010 and 2020
  wdi/wdi_wgi_long.csv                    controls
  ../data/ifsb_share_digitized.csv        Islamic banking share of total banking assets (digitised IFSB charts)

Outputs (code/data):
  panel.csv      one row per country and Findex wave
  barriers.csv   one row per country, wave (2017, 2021) and barrier
"""
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data_raw")
DATA = os.path.join(ROOT, "data")

IFSB_ISO = {
    "Afghanistan": "AFG", "Algeria": "DZA", "Australia": "AUS", "Azerbaijan": "AZE", "Bahrain": "BHR",
    "Bangladesh": "BGD", "Bosnia & Herzegovina": "BIH", "Brunei": "BRN", "Djibouti": "DJI", "Egypt": "EGY",
    "Indonesia": "IDN", "Iran": "IRN", "Iraq": "IRQ", "Jordan": "JOR", "Kazakhstan": "KAZ", "Kenya": "KEN",
    "Kuwait": "KWT", "Kyrgyz": "KGZ", "Kyrgyz Republic": "KGZ", "Lebanon": "LBN", "Libya": "LBY",
    "Malaysia": "MYS", "Maldives": "MDV", "Mauritania": "MRT", "Mauritius": "MUS", "Morocco": "MAR",
    "Nigeria": "NGA", "Oman": "OMN", "Pakistan": "PAK", "Palestine": "PSE", "Philippines": "PHL", "Qatar": "QAT",
    "Saudi Arabia": "SAU", "Senegal": "SEN", "Singapore": "SGP", "Somalia": "SOM", "South Africa": "ZAF",
    "Sri Lanka": "LKA", "Sudan": "SDN", "Switzerland": "CHE", "Tanzania": "TZA", "Thailand": "THA",
    "Tunisia": "TUN", "Turkey": "TUR", "Türkiye": "TUR", "UAE": "ARE", "UK": "GBR", "United Kingdom": "GBR",
    "Yemen": "YEM",
}

PEW_FIX = {
    "Bosnia and Herzegovina": "BIH", "Congo, Democratic Republic": "COD", "Congo, Republic": "COG",
    "East Timor": "TLS", "Ivory Coast": "CIV", "Palestinian territories": "PSE", "Kyrgyzstan": "KGZ",
    "Laos": "LAO", "Micronesia, Federated States": "FSM", "North Korea": "PRK", "South Korea": "KOR",
    "Russia": "RUS", "Syria": "SYR", "Iran": "IRN", "Egypt": "EGY", "Yemen": "YEM", "Gambia": "GMB",
    "Venezuela": "VEN", "Vietnam": "VNM", "Turkey": "TUR", "Slovakia": "SVK", "Czech Republic": "CZE",
    "Cape Verde": "CPV", "Brunei": "BRN", "Macao": "MAC", "Hong Kong": "HKG", "Taiwan": "TWN",
    "St. Lucia": "LCA", "St. Vincent and the Grenadines": "VCT", "Bahamas": "BHS", "Eswatini": "SWZ",
    "Kosovo": "XKX", "Moldova": "MDA", "Tanzania": "TZA", "Bolivia": "BOL", "Somalia": "SOM",
    "United States": "USA", "United Kingdom": "GBR", "North Macedonia": "MKD", "Western Sahara": "ESH",
    "Curacao": "CUW", "Sao Tome and Principe": "STP", "Guinea-Bissau": "GNB", "Libya": "LBY",
}

# reference period of the IFSB chart used for each Findex wave (fieldwork year)
WAVE_TO_IFSB = {2011: "1H2013", 2014: "1H2014", 2017: "1H2017", 2021: "2Q2021", 2022: "2Q2022", 2024: "2Q2024"}
PERIOD_TIME = {"1H2013": 2013.25, "1H2014": 2014.25, "1H2015": 2015.25, "2Q2016": 2016.4, "1H2017": 2017.25,
               "2Q2018": 2018.4, "2Q2019": 2019.4, "2Q2020": 2020.4, "2Q2021": 2021.4, "2Q2022": 2022.4,
               "2Q2024": 2024.4, "3Q2025": 2025.6}

# Steps of at least 5 pp that are not matched by the total assets of the Islamic banking institutions reporting to
# the IFSB PSIFI (results/psifi_steps.csv, column unmatched; psifi_check.py checks that this set agrees with it).
# The share and the PSIFI assets diverge by 15 to 23 % over these steps. PSIFI assets cover the reporting Islamic
# institutions only (in Saudi Arabia four full-fledged banks, without windows), so an unmatched step can be a
# reclassification or growth outside those institutions, such as windows (migration unless they bring in new
# customers); neutralising it is a sensitivity analysis, not a correction.
# The two Bahraini steps (+6.0 pp in 2Q2021, -5.1 pp in 2Q2022) coincide with a fall in the number of reporting
# institutions and are treated together. The Saudi step of 2Q2021 is matched (assets +23 %, share +13 %).
UNMATCHED_STEPS = {("SAU", "2Q2019"), ("KWT", "2Q2021"), ("BHR", "2Q2021"), ("BHR", "2Q2022")}

BARRIERS = {  # keyword that identifies the barrier in both releases (wording differs slightly)
    "distance": "too far away",
    "cost": "too expensive",
    "documents": "necessary documentation",
    "trust": "trust in financial institutions",
    "religion": "religious reasons",
    "funds": "insufficient funds",
    "family": "someone in the family",
}


def findex_usage():
    d = pd.read_csv(os.path.join(RAW, "findex", "GlobalFindexDatabase2025.csv"), low_memory=False)
    d = d[d.group == "all"].copy()
    keep = {
        "account_t_d": "account", "fiaccount_t_d": "fiaccount", "mobileaccount_t_d": "mobileaccount",
        "fin17a": "save_formal", "fin22a": "borrow_formal", "fin22e": "borrow_business",
        "fin22b": "borrow_family", "fin17c": "save_club", "fin22d": "borrow_health", "borrow_any_t_d": "borrow_any",
        "save_any_t_d": "save_any", "inactive_t_d": "inactive", "fin2_t_d": "debitcard",
    }
    cols = ["codewb", "countrynewwb", "year", "regionwb24_hi", "incomegroupwb24", "pop_adult"] + list(keep)
    out = d[cols].rename(columns=keep).rename(columns={"codewb": "iso3", "countrynewwb": "country",
                                                      "regionwb24_hi": "region", "incomegroupwb24": "income"})
    # subgroup accounts for gaps
    g = pd.read_csv(os.path.join(RAW, "findex", "GlobalFindexDatabase2025.csv"), low_memory=False)
    piv = []
    for grp, lab in [("women", "f"), ("men", "m"), ("poorest 40%", "poor"), ("richest 60%", "rich"),
                     ("prim edu or less", "lowedu"), ("secondary edu or more", "hiedu")]:
        s = g[g.group2 == grp][["codewb", "year", "account_t_d", "fin22e", "fin17a", "fin22a"]].rename(
            columns={"codewb": "iso3", "account_t_d": f"account_{lab}", "fin22e": f"borrow_business_{lab}",
                     "fin17a": f"save_formal_{lab}", "fin22a": f"borrow_formal_{lab}"})
        piv.append(s.set_index(["iso3", "year"]))
    sub = pd.concat(piv, axis=1).reset_index()
    out = out.merge(sub, on=["iso3", "year"], how="left")
    return out


def findex_barriers():
    rows = []
    # 2017 release
    d17 = pd.read_excel(os.path.join(RAW, "findex", "Findex2017_Database.xlsx"), sheet_name="Data")
    d17 = d17.rename(columns={"Unnamed: 0": "year", "Unnamed: 1": "iso3", "Unnamed: 2": "country"})
    d17.columns = [c.strip() if isinstance(c, str) else c for c in d17.columns]
    d21 = pd.read_excel(os.path.join(RAW, "findex", "Findex2021_DatabankWide.xlsx"), sheet_name="Data")
    d21 = d21.rename(columns={"Year": "year", "Country code": "iso3", "Country name": "country"})
    d21.columns = [c.strip() if isinstance(c, str) else c for c in d21.columns]
    for src, d, years in [("2017", d17, [2017]), ("2021", d21, [2021, 2022])]:
        d = d[d.year.isin(years)]
        for b, key in BARRIERS.items():
            cc = [c for c in d.columns if isinstance(c, str) and c.startswith("No account") and key in c]
            c_adult = [c for c in cc if "% age 15+" in c]
            c_unb = [c for c in cc if "without" in c]
            if not c_adult or not c_unb:
                raise KeyError((src, b))
            tmp = d[["iso3", "year", c_adult[0], c_unb[0]]].copy()
            tmp.columns = ["iso3", "year", "share_adults", "share_unbanked"]
            tmp["barrier"] = b
            tmp["wave"] = 2017 if src == "2017" else 2021
            rows.append(tmp)
    out = pd.concat(rows).dropna(subset=["share_unbanked"])
    return out


def ifsb_audit():
    """Flag digitised values that are not usable as treatment variation.

    carried_forward: identical value (two decimals, share >= 0.5, below 99.5) in consecutive editions; the later
        value repeats the earlier one rather than adding an observation.
    dup_within: a printed value shared by at least three jurisdictions in the same edition (share >= 0.5); kept,
        and replaced by interpolation in the sensitivity series share_sx.
    outlier: Mauritius 1H2013 (5.47 %) is inconsistent with every later edition (<= 0.2 %).
    step: change of at least 5 pp between consecutive editions (listed in the paper, kept).
    unmatched: the steps in UNMATCHED_STEPS.
    share_ns: share with every step of at least 5 pp neutralised (the step is subtracted from the step period
        onwards, a level shift, so the series keeps all other changes). This deliberately over-neutralises: it also
        removes steps that PSIFI assets match.
    share_np: share with only the unmatched steps neutralised.
    """
    d = pd.read_csv(os.path.join(DATA, "ifsb_share_digitized.csv"))
    d["iso3"] = d.jurisdiction.map(IFSB_ISO)
    assert d.iso3.notna().all(), d[d.iso3.isna()].jurisdiction.unique()
    d["t"] = d.ref_period.map(PERIOD_TIME)
    d = d.sort_values(["iso3", "t"]).reset_index(drop=True)
    prev = d.groupby("iso3").share_pct.shift(1)
    d["carried_forward"] = ((d.share_pct.round(2) == prev.round(2)) & (d.share_pct >= 0.5) & (d.share_pct < 99.5)).astype(int)
    d["outlier"] = ((d.iso3 == "MUS") & (d.ref_period == "1H2013")).astype(int)
    ok = (d.carried_forward == 0) & (d.outlier == 0)
    d["change"] = np.nan
    d.loc[ok, "change"] = d[ok].share_pct - d[ok].groupby("iso3").share_pct.shift(1)
    d["step"] = (d.change.abs() >= 5).astype(int)
    d["unmatched"] = [int((i, p) in UNMATCHED_STEPS) for i, p in zip(d.iso3, d.ref_period)]
    pr = d.groupby(["report", "share_pct"]).iso3.transform("size")
    d["dup_within"] = ((pr >= 3) & (d.share_pct >= 0.5) & (d.share_pct < 99.5)).astype(int)
    for col, flag in [("share_ns", "step"), ("share_np", "unmatched")]:
        jump = np.where((d[flag] == 1) & ok, d.change.fillna(0), 0.0)
        cum = pd.Series(jump).groupby(d.iso3).cumsum()
        d[col] = np.where(d.share_pct >= 99.5, d.share_pct, d.share_pct - cum)
    # sensitivity series: values shared by three or more jurisdictions in one edition replaced by interpolation
    d["share_sx"] = d.share_pct.where(d.dup_within == 0)
    for iso, g in d[ok].groupby("iso3"):
        m = g.share_sx.notna()
        if m.sum() >= 2:
            d.loc[g.index, "share_sx"] = np.interp(g.t, g.t[m], g.share_sx[m])
        else:
            d.loc[g.index, "share_sx"] = g.share_pct
    d.to_csv(os.path.join(DATA, "ifsb_audit.csv"), index=False)
    return d


def ifsb_shares(clean=True):
    d = ifsb_audit()
    if clean:
        d = d[(d.carried_forward == 0) & (d.outlier == 0)]
    return d[["iso3", "ref_period", "t", "share_pct", "share_tick", "share_ref", "share_ns", "share_np", "share_sx"]]


def ib_for_waves(panel_keys, ifsb):
    """Attach the Islamic banking share for each country-wave.

    ib_share: value from the IFSB chart whose reference period matches the Findex wave (WAVE_TO_IFSB); if that
        value was removed by the audit (carried forward or outlier), linear interpolation between the
        neighbouring editions at the matched period.
    ib_interp: linear interpolation in time at mid-year of the survey (no extrapolation beyond one year).
    Morocco had no Islamic bank before May 2017 (first participative bank, IFSB 2018 report), so its share is 0
    in the 2011 and 2014 waves and negligible (0) in the 2017 wave.
    """
    out = []

    def matched(s, per, col):
        v = s[s.ref_period == per][col]
        ib = float(v.iloc[0]) if len(v) else np.nan
        if np.isnan(ib) and per in PERIOD_TIME and len(s) >= 2 and s.t.min() < PERIOD_TIME[per] < s.t.max():
            ib = float(np.interp(PERIOD_TIME[per], s.t, s[col]))
        return ib

    for iso3, year in panel_keys:
        per = WAVE_TO_IFSB.get(year)
        s = ifsb[(ifsb.iso3 == iso3)].sort_values("t")
        ib = matched(s, per, "share_pct")
        ib_ns = matched(s, per, "share_ns")
        ib_np = matched(s, per, "share_np")
        ib_sx = matched(s, per, "share_sx")
        tt = year + 0.5
        if len(s) >= 2 and s.t.min() <= tt <= s.t.max():
            ib_i = float(np.interp(tt, s.t, s.share_pct))
        elif len(s) and abs(s.t - tt).min() <= 1.0:
            ib_i = float(s.iloc[(s.t - tt).abs().argmin()].share_pct)
        else:
            ib_i = np.nan
        if iso3 == "MAR" and year in (2011, 2014, 2017):
            ib, ib_i, ib_ns, ib_np, ib_sx = 0.0, 0.0, 0.0, 0.0, 0.0
        out.append(dict(iso3=iso3, year=year, ib_share=ib, ib_interp=ib_i, ib_ns=ib_ns, ib_np=ib_np, ib_sx=ib_sx,
                        ifsb_listed=int(len(s) > 0)))
    return pd.DataFrame(out)


def pew():
    p = pd.read_csv(os.path.join(RAW, "pew", "pew_religious_composition_2010_2020.csv"))
    p = p[(p.isRegion.astype(str) == "False") & (p.Year.isin([2010, 2020]))]
    wb = pd.read_csv(os.path.join(RAW, "wdi", "wdi_wgi_long.csv"))[["iso3", "country"]].drop_duplicates()
    name2iso = dict(zip(wb.country, wb.iso3))
    p["iso3"] = p.Country.map(lambda c: PEW_FIX.get(c, name2iso.get(c)))
    p = p.dropna(subset=["iso3"])
    w = p.pivot_table(index="iso3", columns="Year", values="Muslims").rename(columns={2010: "muslim2010",
                                                                                     2020: "muslim2020"})
    return w.reset_index()


def wdi():
    w = pd.read_csv(os.path.join(RAW, "wdi", "wdi_wgi_long.csv"))
    w = w.pivot_table(index=["iso3", "year"], columns="var", values="value").reset_index()
    return w


# IMF World Economic Outlook (April 2025) advanced economies; every other economy is an emerging market or
# developing economy (EMDE)
IMF_ADVANCED = {"AND", "AUS", "AUT", "BEL", "CAN", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", "DEU", "GRC",
                "HKG", "ISL", "IRL", "ISR", "ITA", "JPN", "KOR", "LVA", "LTU", "LUX", "MAC", "MLT", "NLD", "NZL",
                "NOR", "PRT", "PRI", "SMR", "SGP", "SVK", "SVN", "ESP", "SWE", "CHE", "TWN", "GBR", "USA"}


def add_flags(df):
    """Sample flags.

    emerging: IMF emerging market or developing economy (not in IMF_ADVANCED).
    developing: World Bank income group (FY2025, as in the Findex 2025 release) other than high income.
    zero_ok: not listed in any IFSB chart and Muslim population share below 1 % in 2020; Islamic banking is
             coded 0 for these comparison countries (ib_z).
    """
    inc = df.groupby("iso3").income.agg(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan)
    df["income_last"] = df.iso3.map(inc)
    df["developing"] = (df.income_last.notna() & (df.income_last != "High income")).astype(int)
    df["emerging"] = (~df.iso3.isin(IMF_ADVANCED)).astype(int)
    meta = pd.read_csv(os.path.join(RAW, "wdi", "country_meta.csv"))
    df["region_geo"] = df.iso3.map(dict(zip(meta.iso3, meta.region_geo)))
    df["zero_ok"] = ((df.ifsb_listed.fillna(0) == 0) & (df.muslim2020 < 1)).astype(int)
    df["ib_z"] = np.where(df.zero_ok == 1, 0.0, df.ib_share)
    df["ib_sys"] = np.where(df.ib_share.notna(), (df.ib_share >= 15).astype(float), np.nan)


def annual_panel(ifsb, panel):
    """Country-year panel 2013-2024: IB share linearly interpolated between IFSB editions (no extrapolation
    beyond half a year from the first or last edition), WDI/FAS outcomes and controls."""
    rows = []
    for iso, g in ifsb.groupby("iso3"):
        g = g.sort_values("t")
        for y in range(2013, 2025):
            tt = y + 0.5
            if g.t.min() - 0.5 <= tt <= g.t.max() + 0.5:
                rows.append((iso, y, float(np.interp(tt, g.t, g.share_pct)), float(np.interp(tt, g.t, g.share_ns)),
                             float(np.interp(tt, g.t, g.share_np)), float(np.interp(tt, g.t, g.share_sx))))
    ib = pd.DataFrame(rows, columns=["iso3", "year", "ib", "ib_ns", "ib_np", "ib_sx"])
    flags = panel.groupby("iso3").agg(emerging=("emerging", "max"), developing=("developing", "max"),
                                      zero_ok=("zero_ok", "max"),
                                      region=("region_geo", "first"), country=("country", "first"),
                                      muslim2020=("muslim2020", "first")).reset_index()
    zero = flags[flags.zero_ok == 1].iso3
    zrows = pd.DataFrame([(i, y, 0.0, 0.0, 0.0, 0.0) for i in zero for y in range(2013, 2025)],
                         columns=["iso3", "year", "ib", "ib_ns", "ib_np", "ib_sx"])
    a = pd.concat([ib, zrows]).merge(flags, on="iso3", how="left")
    w = wdi()
    a = a.merge(w, on=["iso3", "year"], how="left")
    a["lgdppc"] = np.log(a.gdppc_ppp)
    for v in ["depositors", "borrowers"]:
        if v in a:
            a["l" + v] = np.log(a[v])
    a["ifsb_listed"] = (~a.iso3.isin(zero)).astype(int)
    return a


def main():
    usage = findex_usage()
    # one row per country and wave: treat 2022 fieldwork as the 2021 wave
    usage["wave"] = usage.year.replace({2022: 2021})
    usage = usage.sort_values(["iso3", "wave", "year"]).drop_duplicates(["iso3", "wave"], keep="first")
    ifsb = ifsb_shares()
    ib = ib_for_waves(list(zip(usage.iso3, usage.year)), ifsb)
    panel = usage.merge(ib, on=["iso3", "year"], how="left")
    mus = pew()
    panel = panel.merge(mus, on="iso3", how="left")
    # Muslim share interpolated (2010-2020) and held at 2020 afterwards
    panel["muslim"] = np.where(panel.year <= 2020,
                               panel.muslim2010 + (panel.muslim2020 - panel.muslim2010) * (panel.year - 2010) / 10,
                               panel.muslim2020)
    w = wdi()
    panel = panel.merge(w, on=["iso3", "year"], how="left")
    # fill controls from the closest earlier year if the survey year is missing (up to 2 years)
    for c in [c for c in w.columns if c not in ("iso3", "year")]:
        miss = panel[c].isna()
        if miss.any():
            for lag in (1, 2):
                prev = w[["iso3", "year", c]].copy()
                prev["year"] = prev["year"] + lag
                m = panel.loc[miss, ["iso3", "year"]].merge(prev, on=["iso3", "year"], how="left")[c].values
                panel.loc[miss, c] = np.where(np.isnan(panel.loc[miss, c].values.astype(float)), m,
                                              panel.loc[miss, c].values)
                miss = panel[c].isna()
    panel["lgdppc"] = np.log(panel.gdppc_ppp)
    panel["gap_gender"] = panel.account_m - panel.account_f
    panel["gap_income"] = panel.account_rich - panel.account_poor
    add_flags(panel)
    panel.to_csv(os.path.join(DATA, "panel.csv"), index=False)
    annual = annual_panel(ifsb, panel)
    annual.to_csv(os.path.join(DATA, "annual.csv"), index=False)
    print("annual", annual.shape, "with IB:", int(annual.ib.notna().sum()))

    bar = findex_barriers()
    bar = bar.merge(panel[["iso3", "wave", "ib_share", "ib_interp", "ib_ns", "ib_np", "ib_sx", "ib_z", "ifsb_listed", "zero_ok", "emerging",
                           "developing", "region_geo",
                           "muslim", "muslim2020", "account", "lgdppc", "pop_adult", "region", "income", "country"]],
                    on=["iso3", "wave"], how="left")
    bar.to_csv(os.path.join(DATA, "barriers.csv"), index=False)
    print("panel", panel.shape, "IFSB-listed country-waves with share:", int(panel.ib_share.notna().sum()))
    print(panel.groupby("wave").ib_share.count())
    print("barriers", bar.shape)
    print(bar[bar.barrier == "religion"].groupby("wave").agg(n=("iso3", "count"), n_ib=("ib_share", "count")))


if __name__ == "__main__":
    main()
