# IBInclusion: Islamic banking and financial inclusion in emerging economies

Code and data to reproduce the empirical results of the study *Islamic Banking and Financial Inclusion in
Emerging Economies: Bounding the Faith Channel and Testing for Customer Migration*.

The repository contains

* a reproducible panel of **Islamic banking market shares** (share of Islamic banking in total domestic
  banking assets) for 46 jurisdictions (25 to 36 per edition) from twelve editions of the Islamic Financial Services Board
  (IFSB) *Islamic Financial Services Industry Stability Report* (reference periods 1H2013 to 3Q2025): bar
  lengths digitised from the vector charts for the editions that print no values, the printed values for the
  four editions that do, a cross-check of the bar-length method against the printed values and against values
  quoted in the report text, an audit of repeated values and large steps, and a cross-check of the large steps
  against the Islamic banking assets reported to the IFSB Prudential and Structural Islamic Financial Indicators
  (PSIFI) (`data/ifsb_share_digitized.csv`, `data/ifsb_audit.csv`, `results/ifsb_validation*.csv`,
  `results/psifi_*.csv`);
* a four-wave panel of the reasons adults give for having no account at a financial institution (Global Findex
  2011, 2014, 2017 and 2021/22), read from the codebook frequencies that the World Bank Microdata Library publishes
  for every country survey, with a check against the published weighted shares of 2017 and 2021
  (`data/findex_reasons_codebook.csv`, `data/barriers_panel.csv`, `results/codebook_validation.csv`), and the
  interview mode of every survey from the study metadata (`data/findex_survey_mode.csv`);
* the country-wave (Global Findex 2011-2024) and country-year (2013-2024) analysis panels;
* the estimation code for the faith-exclusion bound, the barrier contrast (four-wave panel and the 2017-2021
  difference of the published shares), the implied inclusion and the dose test, the matched outcome contrasts,
  the inclusion-or-migration test, the baseline designs, and the inference procedures (CR1, CR2 with
  Satterthwaite degrees of freedom, CR3 jackknife, wild cluster restricted bootstrap with confidence sets from
  the p-value function, randomisation inference, Holm adjustment, equivalence tests), and the design-based
  simulations (usage outcomes and the barrier panel);
* all estimates as CSV files (`results/`).

## Layout

```
src/fetch_sources.py     download Findex releases, IFSB reports, PSIFI country previews and the Pew table into data_raw/
src/fetch_wdi.py         World Development Indicators, Worldwide Governance Indicators, Financial Access Survey
src/fetch_findex_codebook.py  reasons for having no account, codebook frequencies of all Findex surveys 2011-2021/22
src/digitize_ifsb.py     IFSB share charts: bar lengths (vector rectangles, PyMuPDF) and printed values
src/validate_ifsb.py     bar-length method against printed values and against values quoted in the report text
src/build_panel.py       audit of the share series; data/panel.csv, data/barriers.csv, data/annual.csv
src/psifi_check.py       large steps in the share against PSIFI Islamic banking assets
src/build_barrier_panel.py  four-wave barrier panel from the codebook frequencies; check against the published shares
src/a1_ceiling.py        faith-exclusion bound (two bases), Muslim-specific component, slope robustness
src/a2_barriers.py       barrier contrast on the published shares, first difference 2017-2021
src/a2b_barrier_panel.py barrier contrast on the four-wave panel, offset sensitivity, offset-free measures,
                         precision weighting, Saudi and country-specific trends, interview mode, falsification test,
                         bound by wave, implied reach and account gain (accounting model, with a bound for
                         simultaneous access inclusion), dose test with the 2011 faith pool, share of the Muslim
                         account gap that religious self-exclusion can explain (results/muslim_gap.csv), trend of the
                         religion contrast in the ZERO comparison economies (results/zero_trend.csv); runs after
                         a3_usage.py
src/a3_usage.py          usage and participation: baselines B1-B3, matched contrasts (lambda grid, bracket ends, breakdown), leave-one-out, TWFE weights
src/a3b_gaps.py          components of the income and education account gaps
src/a4_annual.py         inclusion or migration (customer counts, partial-inclusion benchmarks), leads and lags, event study
src/a5_simulation.py     design-based Monte Carlo, power by outcome
src/a5b_barrier_power.py power of the barrier-panel test (binomial sampling of the observed respondents, q = 0 and
                         q = 1), reach detected with 80% power
src/common.py            fixed-effects OLS (FWL), CR1, CR2, CR3, effective clusters, wild cluster restricted bootstrap, RI, Holm
run_all.py               runs the pipeline and records the runtime of each step (results/runtime.csv)
data/                    derived data (digitised shares, audit, panels)
results/                 all estimates as CSV
```

## Reproduce

```bash
pip install -r requirements.txt
python src/fetch_sources.py      # once; IFSB and Internet Archive hosts can be slow
python src/fetch_wdi.py          # once; World Bank API
python src/fetch_findex_codebook.py   # once; World Bank Microdata Library catalogue API (about 20 min)
python run_all.py                # see results/runtime.csv for the time of each step (99,999 bootstrap draws)
```

Raw source files are not redistributed here: the IFSB reports remain the property of the IFSB, and the
Findex and World Bank data are available from the World Bank under CC BY 4.0. `fetch_sources.py`
lists the exact URLs. All random procedures use fixed seeds (bootstrap and permutation: seed 0;
simulation: seeds 0-4 and 100-104; barrier-panel power: seeds 0 and 1).

## Data sources

* World Bank, Global Findex Database (2017, 2021 and 2025 releases), and the codebooks of the Global Findex
  country surveys of 2011, 2014, 2017 and 2021/22 in the World Bank Microdata Library (frequencies only).
* Islamic Financial Services Board, Islamic Financial Services Industry Stability Report, editions 2014-2023,
  2025 and 2026; PSIFI country data previews.
* Pew Research Center, Dataset of Global Religious Composition Estimates for 2010 and 2020 (2025), doi:10.58094/vhrw-k516.
* World Bank, World Development Indicators and Worldwide Governance Indicators; IMF Financial Access Survey; World Bank
  Global Financial Development Database (September 2022 version, indicator GFDD.DI.02, used only for psifi_check.py).

## Model weights

The study is econometric: every estimate is computed from the data in `data/` by the scripts in `src/`, and
there are no trained model weights to distribute.

## Licence

MIT (code). Derived data are provided for verification of the results; cite the original sources.
