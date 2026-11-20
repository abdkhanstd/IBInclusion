"""Digitise the IFSB chart 'Islamic Banking Share in Total Banking Assets by Jurisdiction'.

Each IFSI Stability Report (IFSB) shows the domestic market share of Islamic banking by jurisdiction as a
horizontal bar chart stored as vector rectangles. For every bar we read its left and right edge from the PDF
drawing commands and convert the length to a percentage in two independent ways:

  (a) tick calibration: a linear fit of the axis tick values on the tick label centres;
  (b) reference calibration: the Iran bar is 100 % (Iran has operated a fully Islamic banking system by
      law since 1983), so share = 100 * (x1 - x0) / (x1_Iran - x0).

Labels are attached by vertical proximity to the jurisdiction names. In the 2016 and 2018 editions the names
are outlined graphics rather than text, so the label order was transcribed from a 400 dpi render of the chart
(the label count is checked against the bar count).

Outputs (code/data):
  ifsb_share_digitized.csv   report, ref_period, jurisdiction, share_tick, share_ref, share_pct
  ifsb_digitize_info.csv     calibration diagnostics per report
  ifsb_charts/chart_<year>.png  render of each chart for visual checking

The 2022, 2023, 2025 and 2026 editions print the value at the end of each bar. These printed values are read
as text (share_printed) and used as share_pct for those editions; bar lengths are used for the earlier
editions, which print no values. Where both exist, the printed values check the bar-length method: in 2025 and
2026 the two agree closely, the 2022 bars are drawn on a coarse grid, and the 2023 chart reuses the bar
graphics of the 2022 chart with updated printed values (so its bar lengths are stale).
"""
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pymupdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data_raw", "ifsb")
OUT = os.path.join(ROOT, "data")

# report year -> (page index, reference period of the data in the chart)
CHARTS = {
    2014: (29, "1H2013"),
    2015: (23, "1H2014"),
    2016: (25, "1H2015"),
    2017: (20, "2Q2016"),
    2018: (27, "1H2017"),
    2019: (23, "2Q2018"),
    2020: (18, "2Q2019"),
    2021: (18, "2Q2020"),
    2022: (24, "2Q2021"),
    2023: (24, "2Q2022"),
    2025: (14, "2Q2024"),
    2026: (17, "3Q2025"),
}

# label order transcribed from the rendered chart (names are outlined graphics in 2016 and 2018,
# and in 2014 the tightly spaced text lines cannot be matched reliably)
MANUAL_LABELS = {
    2014: ["Iran", "Sudan", "Saudi Arabia", "Kuwait", "Yemen", "Brunei", "Qatar", "Malaysia", "UAE", "Bangladesh",
           "Bahrain", "Jordan", "Pakistan", "Mauritius", "Turkey", "Egypt", "Indonesia", "Lebanon", "Thailand",
           "Switzerland", "South Africa", "Australia", "UK", "Singapore", "Philippines"],
    2016: ["Iran", "Sudan", "Brunei", "Saudi Arabia", "Kuwait", "Yemen", "Qatar", "Malaysia", "Bangladesh", "UAE",
           "Djibouti", "Jordan", "Bahrain", "Pakistan", "Palestine", "Egypt", "Oman", "Turkey", "Indonesia",
           "Tunisia", "Kenya", "Azerbaijan", "Algeria", "South Africa", "Sri Lanka", "Thailand", "Lebanon",
           "Nigeria", "UK", "Singapore", "Mauritius"],
    2018: ["Iran", "Sudan", "Brunei", "Saudi Arabia", "Kuwait", "Qatar", "Malaysia", "UAE", "Bangladesh",
           "Djibouti", "Jordan", "Bahrain", "Palestine", "Pakistan", "Oman", "Egypt", "Maldives", "Afghanistan",
           "Indonesia", "Turkey", "Senegal", "Tunisia", "Bosnia & Herzegovina", "Algeria", "Kyrgyz", "Iraq",
           "Kenya", "Tanzania", "Lebanon", "Sri Lanka", "Thailand", "Nigeria", "South Africa", "Kazakhstan", "UK",
           "Mauritius"],
}


def filled_rects(page):
    rects = []
    for d in page.get_drawings():
        if d.get("fill") is None:
            continue
        for it in d["items"]:
            if it[0] == "re":
                rects.append(pymupdf.Rect(it[1]))
            elif it[0] == "qu":
                rects.append(it[1].rect)
    return rects


def pick_bars(rects):
    cand = [r for r in rects if 1.5 < r.height < 12 and r.width < 450]
    c = Counter(round(r.x0) for r in cand)
    best, best_score = None, -1.0
    for x0c, n in c.items():
        if n < 15:
            continue
        w = [r.width for r in cand if round(r.x0) == x0c]
        score = n * float(np.std(w))
        if score > best_score:
            best, best_score = x0c, score
    bars = [r for r in cand if abs(r.x0 - best) <= 1.5]
    bars.sort(key=lambda r: r.y0)
    return bars


def axis_ticks(page, bars):
    """Numeric tick labels in one row above or below the bars: list of (value, x_centre)."""
    x0 = min(b.x0 for b in bars)
    x1 = max(b.x1 for b in bars)
    y_top = min(b.y0 for b in bars)
    y_bot = max(b.y1 for b in bars)
    words = page.get_text("words")
    cand = []
    for w in words:
        t = w[4].strip()
        yc = (w[1] + w[3]) / 2
        if not (x0 - 20 <= w[0] <= x1 + 40):
            continue
        if not (y_top - 30 <= yc <= y_top or y_bot <= yc <= y_bot + 30):
            continue
        # a word may hold several concatenated ticks such as 0%10%20%: split on %
        parts = [p for p in re.split(r"%", t) if p != ""]
        if not parts or not all(re.fullmatch(r"\d{1,3}(\.\d)?", p) for p in parts):
            continue
        if len(parts) == 1:
            cand.append((float(parts[0]), (w[0] + w[2]) / 2, round(yc)))
        else:
            # concatenated labels: use character boxes
            cand += split_concatenated(page, w, round(yc))
    if not cand:
        return []
    rows = Counter(c[2] for c in cand)
    yrow, n = rows.most_common(1)[0]
    row = sorted([(v, x) for v, x, y in cand if abs(y - yrow) <= 2], key=lambda t: t[1])
    vals = [v for v, _ in row]
    if len(row) < 5 or any(b <= a for a, b in zip(vals, vals[1:])):
        return []
    return row


def split_concatenated(page, w, yrow):
    out = []
    raw = page.get_text("rawdict", clip=pymupdf.Rect(w[:4]))
    chars = [ch for b in raw["blocks"] for l in b.get("lines", []) for s in l["spans"] for ch in s["chars"]]
    buf = []
    for ch in chars + [{"c": "%", "bbox": (0, 0, 0, 0)}]:
        if ch["c"] == "%":
            if buf:
                v = float("".join(c["c"] for c in buf))
                xc = (buf[0]["bbox"][0] + buf[-1]["bbox"][2]) / 2
                out.append((v, xc, yrow))
            buf = []
        elif ch["c"].isdigit() or ch["c"] == ".":
            buf.append(ch)
    return out


def text_labels(page, bars):
    x0 = min(b.x0 for b in bars)
    y_top = min(b.y0 for b in bars) - 6
    y_bot = max(b.y1 for b in bars) + 6
    lines = {}
    for w in page.get_text("words"):
        yc = (w[1] + w[3]) / 2
        if w[2] <= x0 + 1 and w[0] > x0 - 140 and y_top <= yc <= y_bot:
            lines.setdefault((w[5], w[6]), []).append((w[0], yc, w[4], w[2]))
    out = []
    for ws in lines.values():
        ws.sort()
        # names are right-aligned against the axis: drop text lines that end far from it
        if max(w[3] for w in ws) < x0 - 15:
            continue
        out.append((float(np.mean([w[1] for w in ws])), " ".join(w[2] for w in ws)))
    out.sort()
    return out


def printed_values(page, bars):
    """Values printed just right of the bar ends (2026 edition): {round(bar centre, 1): value}."""
    out = {}
    words = page.get_text("words")
    for b in bars:
        yc = (b.y0 + b.y1) / 2
        best = None
        for w in words:
            m = re.fullmatch(r"(\d{1,3}\.\d)%", w[4].strip())
            wy = (w[1] + w[3]) / 2
            if m and abs(wy - yc) < 3 and b.x1 - 3 <= w[0] <= b.x1 + 15:
                if best is None or w[0] < best[0]:
                    best = (w[0], float(m.group(1)))
        if best is not None:
            out[round(yc, 1)] = best[1]
    return out


def digitize(year):
    pno, period = CHARTS[year]
    page = pymupdf.open(os.path.join(RAW, f"IFSI_SR_{year}.pdf"))[pno]
    bars = pick_bars(filled_rects(page))
    if year in MANUAL_LABELS:
        names = MANUAL_LABELS[year]
        assert len(names) == len(bars), (year, len(names), len(bars))
        labels = [((b.y0 + b.y1) / 2, n) for b, n in zip(bars, names)]
    else:
        labels = text_labels(page, bars)
    ticks = axis_ticks(page, bars)
    printed = printed_values(page, bars)
    rows = []
    for b in bars:
        yc = (b.y0 + b.y1) / 2
        j = int(np.argmin([abs(l[0] - yc) for l in labels]))
        rows.append(dict(report=year, ref_period=period, page=pno + 1, jurisdiction=labels[j][1],
                         label_dist=round(abs(labels[j][0] - yc), 2), x0=b.x0, x1=b.x1, yc=yc,
                         share_printed=printed.get(round(yc, 1), np.nan)))
    df = pd.DataFrame(rows)
    # (b) reference calibration on Iran
    iran = df[df.jurisdiction.str.strip() == "Iran"]
    xref = float(iran.x1.max()) if len(iran) else np.nan
    df["share_ref"] = 100 * (df.x1 - df.x0) / (xref - df.x0)
    # (a) tick calibration
    if ticks:
        v = np.array([t[0] for t in ticks])
        x = np.array([t[1] for t in ticks])
        slope, icpt = np.polyfit(x, v, 1)
        resid = float(np.max(np.abs(v - (slope * x + icpt))))
        df["share_tick"] = slope * df.x1 + icpt - (slope * df.x0 + icpt)
    else:
        slope = icpt = resid = np.nan
        df["share_tick"] = np.nan
    df["share_pct"] = df["share_ref"].where(df["share_ref"].notna(), df["share_tick"])
    os.makedirs(os.path.join(OUT, "ifsb_charts"), exist_ok=True)
    clip = pymupdf.Rect(min(b.x0 for b in bars) - 140, min(b.y0 for b in bars) - 30,
                        max(b.x1 for b in bars) + 30, max(b.y1 for b in bars) + 20)
    page.get_pixmap(clip=clip, dpi=150).save(os.path.join(OUT, "ifsb_charts", f"chart_{year}.png"))
    dup = df.jurisdiction.duplicated().sum()
    info = dict(report=year, ref_period=period, n_bars=len(bars), n_labels=len(labels), n_dup_labels=int(dup),
                max_label_dist=float(df.label_dist.max()), n_ticks=len(ticks), tick_fit_maxresid=resid,
                iran_present=bool(len(iran)),
                mean_abs_tick_minus_ref=float((df.share_tick - df.share_ref).abs().mean()) if ticks else np.nan,
                max_abs_tick_minus_ref=float((df.share_tick - df.share_ref).abs().max()) if ticks else np.nan,
                n_printed=int(df.share_printed.notna().sum()),
                mean_abs_ref_minus_printed=float((df.share_ref - df.share_printed).abs().mean()),
                max_abs_ref_minus_printed=float((df.share_ref - df.share_printed).abs().max()))
    return df, info


def main():
    years = [int(a) for a in sys.argv[1:]] or sorted(CHARTS)
    dfs, infos = [], []
    for y in years:
        df, info = digitize(y)
        dfs.append(df)
        infos.append(info)
        print(info)
    out = pd.concat(dfs)
    # a bar drawn as two stacked rectangles (two colours) appears twice with the same label and length
    out = out.drop_duplicates(subset=["report", "jurisdiction", "x1"])
    out["share_ref"] = out["share_ref"].round(2)
    out["share_tick"] = out["share_tick"].round(2)
    out["share_bar"] = out["share_pct"].round(2)
    out["share_pct"] = out["share_printed"].where(out["share_printed"].notna(), out["share_bar"]).round(2)
    out["source"] = np.where(out["share_printed"].notna(), "printed", "bar")
    out[["report", "ref_period", "page", "jurisdiction", "label_dist", "share_tick", "share_ref", "share_bar",
         "share_printed", "share_pct", "source"]].to_csv(
        os.path.join(OUT, "ifsb_share_digitized.csv"), index=False)
    pd.DataFrame(infos).to_csv(os.path.join(OUT, "ifsb_digitize_info.csv"), index=False)


if __name__ == "__main__":
    main()
