"""Reasons for not having an account, all four Findex waves that asked about religion (2011, 2014, 2017, 2021).

The country-level Findex releases publish the reasons only for 2017 and 2021. The 2011 and 2014 surveys asked
the same question (each reason a separate yes/no item, asked to adults without an account at a financial
institution), and the World Bank Microdata Library publishes, for every country survey, the codebook frequency
of each answer category. This script reads those codebook frequencies through the public catalogue API
(https://microdata.worldbank.org/index.php/api/catalog/...) for every Findex survey of 2011, 2014, 2017 and
2021/22 and writes

  data_raw/findex/codebook/<idno>.json        raw API responses (cache)
  data_raw/findex/codebook/mode_<idno>.json   mode of data collection from the study metadata (cache)
  data/findex_reasons_codebook.csv            iso3, wave, survey_year, idno, barrier, yes, no, dk, ref
  data/findex_survey_mode.csv                 iso3, wave, survey_year, idno, mode (as recorded), phone (0/1)

The frequencies are unweighted respondent counts. build_barrier_panel.py turns them into shares of the adults
asked and compares them with the published weighted shares of 2017 and 2021.
"""
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data_raw", "findex", "codebook")
OUT = os.path.join(ROOT, "data", "findex_reasons_codebook.csv")
API = "https://microdata.worldbank.org/index.php/api/catalog"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research data client"}
WAVES = {2011: 2011, 2014: 2014, 2017: 2017, 2021: 2021, 2022: 2021}
KEYS = [("religion", "religio"), ("distance", "far"), ("cost", "expensive"), ("documents", "documentation"),
        ("trust", "trust"), ("funds", "money"), ("family", "family"), ("noneed", "no need"),
        ("cannot", "cannot get")]


def get(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode("utf8"))
        except Exception:  # transient network errors: back off and retry
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))


def studies():
    path = os.path.join(RAW, "_studies.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf8"))
    rows = []
    for pg in range(1, 20):
        d = get(f"{API}/search?sk=findex&ps=100&page={pg}")["result"]
        rows += d["rows"]
        if len(d["rows"]) < 100:
            break
    rows = [{k: r[k] for k in ("id", "idno", "nation", "year_start")} for r in rows
            if r.get("repositoryid") == "global-findex" and r["year_start"] in WAVES]
    os.makedirs(RAW, exist_ok=True)
    json.dump(rows, open(path, "w", encoding="utf8"))
    return rows


def reason_vars(idno):
    out, page = [], 1
    while True:
        d = get(f"{API}/{idno}/variables?limit=100&page={page}")
        out += d["variables"]
        if len(out) >= int(d["total"]) or not d["variables"]:
            break
        page += 1
    sel = []
    for v in out:
        lab = (v.get("labl") or "").lower()
        if ("does not have account" in lab or "reason for no account" in lab) and "mobile" not in lab:
            for b, key in KEYS:
                if key in lab:
                    sel.append((b, v["vid"], v["name"], v.get("labl")))
                    break
    return sel


def study(s):
    path = os.path.join(RAW, f"{s['idno']}.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf8"))
    rec = {"idno": s["idno"], "year": s["year_start"], "vars": []}
    for b, vid, name, lab in reason_vars(s["idno"]):
        meta = get(f"{API}/{s['idno']}/variables/{vid}")["variable"]["metadata"]
        cats = {}
        for c in meta.get("var_catgry") or []:
            f = [x for x in (c.get("stats") or []) if x.get("type") == "freq" and not x.get("wgtd")]
            cats[str(c.get("labl")).strip().lower()] = int(float(f[0]["value"])) if f else None
        rec["vars"].append({"barrier": b, "vid": vid, "name": name, "label": lab, "cats": cats})
        time.sleep(0.15)
    json.dump(rec, open(path, "w", encoding="utf8"))
    return rec


def study_mode(s):
    """Mode of data collection recorded in the study metadata (method / data_collection / coll_mode)."""
    path = os.path.join(RAW, f"mode_{s['idno']}.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf8"))
    d = get(f"{API}/{s['idno']}")
    found = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "coll_mode":
                    found.append(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(d)
    vals = []
    for v in found:
        vals += v if isinstance(v, list) else [v]
    rec = {"idno": s["idno"], "coll_mode": " | ".join(str(v).strip() for v in vals if v)}
    json.dump(rec, open(path, "w", encoding="utf8"))
    return rec


def modes(ss):
    with ThreadPoolExecutor(max_workers=8) as ex:
        recs = list(ex.map(study_mode, ss))
    rows = []
    for s, r in zip(ss, recs):
        m = r["coll_mode"]
        ml = m.lower()
        phone = int(("telephone" in ml or "phone" in ml or "cati" in ml) and "face" not in ml and "capi" not in ml
                    and "papi" not in ml)
        rows.append({"iso3": s["idno"][:3], "wave": WAVES[s["year_start"]], "survey_year": s["year_start"],
                     "idno": s["idno"], "mode": m, "phone": phone})
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(ROOT, "data", "findex_survey_mode.csv"), index=False)
    print(d.groupby("wave").phone.agg(["count", "sum"]))


def main():
    os.makedirs(RAW, exist_ok=True)
    rows = []
    ss = studies()
    print(len(ss), "studies")

    def safe(s):
        try:
            return study(s)
        except Exception as e:  # report and continue; rerun fills the gaps from the cache
            print("FAILED", s["idno"], e, flush=True)
            return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        recs = list(ex.map(safe, ss))
    for i, (s, rec) in enumerate(zip(ss, recs)):
        if rec is None:
            continue
        for v in rec["vars"]:
            c = v["cats"]
            rows.append({"iso3": s["idno"][:3], "wave": WAVES[s["year_start"]], "survey_year": s["year_start"],
                         "idno": s["idno"], "barrier": v["barrier"], "var": v["name"],
                         "yes": c.get("yes"), "no": c.get("no"), "dk": c.get("(dk)", c.get("dk")),
                         "ref": c.get("(ref)", c.get("refused"))})
        if i % 25 == 0:
            print(i, s["idno"], len(rec["vars"]), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    print(d.groupby(["wave", "barrier"]).size().unstack())
    modes(ss)


if __name__ == "__main__":
    main()
