# Aggregate On-Chain Liquidity Indicators and Future DeFi Stress

**Author:** Yifan Zhi  
**Course:** STAT8308 Blockchain data analytics [Section SA, 2025]

This repository contains the code, data tables, model outputs, figures, and final report for the STAT8308 Applied Web3 Data Science Capstone.

## Research Question

Do stablecoin contraction, common TVL-return movement, downside breadth, and realised volatility add out-of-sample information about a future decline in a selected cross-protocol DeFi liquidity basket?

The primary target is a decline of at least 5% in the equal-weight USD TVL basket over days `t+1` through `t+7`, using predictors observed through day `t`. USD TVL combines token-price and token-quantity changes, so the analysis concerns USD TVL stress rather than pure depositor flows.

## Main Findings

- The final chronological holdout contains 465 observations and 93 stress labels; event prevalence is 0.200.
- The prespecified five-component score has holdout average precision of 0.153.
- Stablecoin contraction alone reaches 0.267 and is the strongest prespecified signal.
- A chronologically selected inverse-correlation model reaches 0.396, but feature choice and performance vary sharply across folds and target definitions.
- ETH adjustment reduces mean holdout pairwise TVL-return correlation from 0.619 to 0.105.
- Stablecoin average precision falls from 0.267 at `t` to 0.204 at `t-7`, leaving little evidence of a dependable one-week-ahead warning.

The results do not establish direct economic contagion. The correlation graph measures simultaneous statistical co-movement, much of which reflects common ETH exposure.

## Repository Structure

```text
.
|-- README.md
|-- CHANGELOG.md
|-- requirements.txt
|-- run_pipeline.sh
|-- sql/
|   `-- build_features.sql
|-- src/
|   |-- config.py
|   |-- 01_fetch_defillama.py
|   |-- 02_build_features.py
|   |-- 02b_build_market_control.py
|   |-- 03_model_contagion.py
|   |-- model_liquidity_stress.py
|   |-- 04_make_visuals_and_report.py
|   |-- 05_make_academic_report.py
|   `-- 06_validate_outputs.py
|-- data/
|   |-- raw/          # endpoint manifests and small ETH source snapshots
|   |-- staging/      # normalized source tables used for the submitted analysis
|   `-- processed/    # model-ready data and market control
|-- models/           # machine-readable results and validation receipt
|-- figures/          # eight publication-ready figures
`-- report/
    |-- technical_report.md
    `-- technical_report.pdf
```

Internal audit and review files are intentionally excluded from this repository.

## Reproduce the Submitted Results

Python 3.10 or newer is recommended.

```bash
python3 -m pip install -r requirements.txt
REPRODUCE_SUBMITTED_RESULTS=1 ./run_pipeline.sh
```

This mode uses the committed staging and processed tables and rebuilds all model outputs, figures, the report, and the quality-control receipt without internet access.

## Run a Fresh Extraction

```bash
./run_pipeline.sh
```

The default mode downloads protocol TVL and aggregate stablecoin supply from the documented DefiLlama endpoints, rebuilds the analytical data, and then runs the complete analysis. No API key is required. Fresh source data can differ from the submitted snapshot because public API histories may be extended or revised.

Exact endpoints used for the submitted snapshot:

- `https://api.llama.fi/protocol/{slug}`
- `https://stablecoins.llama.fi/stablecoincharts/all`
- `https://coins.llama.fi/chart/coingecko:ethereum?start={start}&span=500&period=1d`

## Empirical Design

- Manually selected eight-protocol sample across lending, DEX, liquid staking, CDP, and yield categories.
- Fixed return bounds of -0.35 and 0.35; missing pre-launch observations are not filled with zero.
- 60-day pairwise-complete correlation network and a 0.45 density threshold.
- Protocol-specific ETH factor adjustment using betas estimated from days `t-180` through `t-1`.
- Purged 70/30 chronological holdout plus four expanding walk-forward folds.
- Fold-local scaling, feature selection, coefficient estimation, and operating thresholds.
- Equal-weight, lagged-TVL-weighted, capped, category-balanced, and common-history targets.
- Average precision, ROC area, component redundancy, ablation, lag, episode, and moving-block bootstrap diagnostics.
- Conditional scenario grid across 64 propagation parameter combinations.

## Key Files

- `report/technical_report.pdf`: final eight-page academic report
- `models/validation_metrics.csv`: final holdout comparison
- `models/walk_forward_metrics.csv`: fold and aggregate out-of-fold results
- `models/target_definition_sensitivity.csv`: alternative basket targets
- `models/eth_factor_network_summary.csv`: raw versus ETH-adjusted network statistics
- `models/component_ablation_incremental.csv`: single-factor and incremental tests
- `models/lag_horizon_performance.csv`: 3-, 7-, and 14-day outcomes at multiple predictor lags
- `models/episode_details.csv`: episode onset, lead time, duration, severity, and protocol drivers
- `models/scenario_parameter_sensitivity.csv`: conditional scenario grid
- `models/quality_control.json`: automated 22-check validation receipt

## Submission

The assignment submission consists of the final PDF and the public GitHub repository URL. The code does not need to be uploaded separately outside GitHub unless the course submission page explicitly requests an additional archive.
