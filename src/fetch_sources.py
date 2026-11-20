"""Download the public source files into data_raw/ (run once before run_all.py).

  findex/GlobalFindexDatabase2025.csv     World Bank, Global Findex 2025 release (CC BY 4.0)
  findex/Findex2021_DatabankWide.xlsx     World Bank, Global Findex 2021 release (barriers 2021/22)
  findex/Findex2017_Database.xlsx         World Bank, Global Findex 2017 release (barriers 2017), via the
                                          Internet Archive copy of the former download page
  ifsb/IFSI_SR_<year>.pdf                 IFSB Islamic Financial Services Industry Stability Reports (2013-2023, 2025, 2026)
  ifsb/psifi/<country>.pdf                IFSB PSIFI country data previews (Islamic banking structural indicators)
  pew/pew_religious_composition_2010_2020.csv   table embedded in the Pew Research Center feature page
WDI, WGI and Financial Access Survey series are fetched by fetch_wdi.py.
"""
import json
import os
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data_raw")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0 Safari/537.36"}

FILES = {
    "findex/GlobalFindexDatabase2025.csv":
        "https://thedocs.worldbank.org/en/doc/be6615202d1f08a25855c8ac2d615122-0050012025/related/"
        "GlobalFindexDatabase2025.csv",
    "findex/Findex2021_DatabankWide.xlsx":
        "https://thedocs.worldbank.org/en/doc/6d8435d479acefaa8960dc85f47efb6a-0430062023/original/DatabankWide.xlsx",
    "findex/Findex2017_Database.xlsx":
        "https://web.archive.org/web/20210601165113id_/https://globalfindex.worldbank.org/sites/globalfindex/files/"
        "databank/Global%20Findex%20Database.xlsx",
    "ifsb/IFSI_SR_2025.pdf": "https://www.ifsb.org/wp-content/uploads/2025/05/IFSI-Stability-Report-May-2025.pdf",
    "ifsb/IFSI_SR_2026.pdf": "https://www.ifsb.org/wp-content/uploads/2026/05/IFSR-2026.pdf",
}
# PSIFI country data previews (structural indicators of the reporting Islamic banking institutions)
for c in ["Saudi-Arabia", "Kuwait", "Bahrain", "Malaysia", "Pakistan", "Bangladesh", "Egypt", "Jordan"]:
    FILES[f"ifsb/psifi/{c}.pdf"] = f"https://www.ifsb.org/wp-content/uploads/2023/10/{c}_Islamic-Banking-Data_Preview_En.pdf"
for y in range(2013, 2024):
    FILES[f"ifsb/IFSI_SR_{y}.pdf"] = (f"https://www.ifsb.org/wp-content/uploads/2023/10/"
                                       f"Islamic-Financial-Services-Industry-Stability-Report-{y}_En.pdf")
PEW_URL = "https://www.pewresearch.org/religion/feature/religious-composition-by-country-2010-2020/"


def get(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as fh:
        fh.write(r.read())
    print("downloaded", path)


def pew(path):
    if os.path.exists(path):
        return
    req = urllib.request.Request(PEW_URL, headers=UA)
    t = urllib.request.urlopen(req, timeout=120).read().decode("utf8", "ignore")
    i = t.find('"rows":[{"Region')
    start = t.find("[", i)
    depth = 0
    for j in range(start, len(t)):
        if t[j] == "[":
            depth += 1
        elif t[j] == "]":
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    rows = json.loads(t[start:end])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print("extracted", path)


def main():
    for rel, url in FILES.items():
        try:
            get(url, os.path.join(RAW, rel))
        except Exception as e:  # report and continue; some hosts are intermittently unavailable
            print("FAILED", rel, e)
    pew(os.path.join(RAW, "pew", "pew_religious_composition_2010_2020.csv"))


if __name__ == "__main__":
    main()
