"""Download country-year controls from the World Bank API (WDI source 2, WGI source 3)."""
import json
import os
import time
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data_raw", "wdi")
os.makedirs(OUT, exist_ok=True)

SERIES = {
    2: {
        "NY.GDP.PCAP.PP.KD": "gdppc_ppp",
        "SP.POP.TOTL": "pop",
        "SP.URB.TOTL.IN.ZS": "urban",
        "IT.CEL.SETS.P2": "mobile",
        "IT.NET.USER.ZS": "internet",
        "FS.AST.PRVT.GD.ZS": "privcredit",
        "FB.CBK.BRCH.P5": "branches",
        "FB.ATM.TOTL.P5": "atms",
        "SL.TLF.CACT.FE.ZS": "lfp_female",
        "SL.TLF.CACT.MA.ZS": "lfp_male",
        "SL.TLF.CACT.ZS": "lfp_total",
        "SL.EMP.SELF.ZS": "selfemp",
        "SL.EMP.SELF.FE.ZS": "selfemp_female",
        "SE.SEC.ENRR": "sec_enrol",
        "SP.POP.1564.TO.ZS": "pop1564",
        "NY.GDP.MKTP.KD.ZG": "gdp_growth",
        "FP.CPI.TOTL.ZG": "inflation",
        "SI.POV.DDAY": "poverty215",
        "FB.CBK.DPTR.P3": "depositors",
        "FB.CBK.BRWR.P3": "borrowers",
        "FS.AST.DOMS.GD.ZS": "domcredit",
    },
    3: {
        "GOV_WGI_RL.EST": "rule_of_law",
        "GOV_WGI_RQ.EST": "reg_quality",
        "GOV_WGI_CC.EST": "control_corruption",
    },
}


def fetch(source, code):
    cache = os.path.join(OUT, f"raw_{code}.json")
    if os.path.exists(cache):
        with open(cache) as fh:
            return json.load(fh)
    url = (f"https://api.worldbank.org/v2/country/all/indicator/{code}?source={source}"
           f"&format=json&per_page=20000&date=2008:2025")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                d = json.loads(r.read().decode())
            with open(cache, "w") as fh:
                json.dump(d, fh)
            return d
        except Exception as e:  # network is unreliable: retry
            print("retry", code, e)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(code)


def main():
    frames = []
    for source, series in SERIES.items():
        for code, name in series.items():
            d = fetch(source, code)
            rows = [dict(iso3=r["countryiso3code"], country=r["country"]["value"], year=int(r["date"]),
                         var=name, value=r["value"]) for r in d[1] if r["value"] is not None]
            print(code, name, len(rows), flush=True)
            frames.append(pd.DataFrame(rows))
    df = pd.concat(frames)
    df.to_csv(os.path.join(OUT, "wdi_wgi_long.csv"), index=False)


if __name__ == "__main__":
    main()
