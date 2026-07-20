from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, REPORT_DIR, ensure_directories


TITLE = "Aggregate On-Chain Liquidity Indicators and Future DeFi Stress"
SUBTITLE = "An Out-of-Sample Evaluation"
AUTHOR = "Yifan Zhi"
COURSE = "STAT8308 Blockchain data analytics [Section SA, 2025]"
DATE = "July 2026"
REPOSITORY = "https://github.com/Zzhi993/defi-liquidity-stress-monitor"


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def load_context() -> dict[str, Any]:
    def csv(name: str) -> pd.DataFrame:
        return pd.read_csv(MODELS_DIR / name)

    with open(MODELS_DIR / "model_summary.json", encoding="utf-8") as handle:
        summary = json.load(handle)
    with open(MODELS_DIR / "calibration_parameters.json", encoding="utf-8") as handle:
        calibration = json.load(handle)
    with open(PROCESSED_DIR.parent / "staging" / "source_manifest.json", encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    with open(PROCESSED_DIR.parent / "raw" / "market_manifest.json", encoding="utf-8") as handle:
        market_manifest = json.load(handle)

    return {
        "summary": summary,
        "calibration": calibration,
        "source_manifest": source_manifest,
        "market_manifest": market_manifest,
        "coverage": csv("protocol_coverage.csv"),
        "validation": csv("validation_metrics.csv"),
        "walk": csv("walk_forward_metrics.csv"),
        "target": csv("target_definition_sensitivity.csv"),
        "vif": csv("component_vif.csv"),
        "associations": csv("component_associations.csv"),
        "ablation": csv("component_ablation_incremental.csv"),
        "incremental": csv("incremental_logistic_tests.csv"),
        "coefficients": csv("component_coefficient_stability.csv"),
        "factor_network": csv("eth_factor_network_summary.csv"),
        "factor_models": csv("eth_factor_model_comparison.csv"),
        "scaling": csv("scaling_method_comparison.csv"),
        "episodes": csv("episode_details.csv"),
        "event_study": csv("episode_event_study.csv"),
        "scenario": csv("scenario_parameter_sensitivity.csv"),
        "scenario_historical": csv("scenario_historical_comparison.csv"),
        "scenario_assumptions": csv("scenario_assumptions.csv"),
        "lag": csv("lag_horizon_performance.csv"),
        "selection": csv("final_training_model_selection.csv"),
    }


def section(title: str) -> dict[str, Any]:
    return {"kind": "section", "title": title}


def paragraph(text: str) -> dict[str, Any]:
    return {"kind": "paragraph", "text": text}


def figure(number: int, title: str, filename: str, width: float = 5.55) -> dict[str, Any]:
    return {
        "kind": "figure",
        "number": number,
        "title": title,
        "filename": filename,
        "width": width,
    }


def table(number: int, title: str, headers: list[str], rows: list[list[str]], widths: list[float]) -> dict[str, Any]:
    return {
        "kind": "table",
        "number": number,
        "title": title,
        "headers": headers,
        "rows": rows,
        "widths": widths,
    }


def reference(text: str) -> dict[str, Any]:
    return {"kind": "reference", "text": text}


def code(text: str) -> dict[str, Any]:
    return {"kind": "code", "text": text}


def build_blocks(c: dict[str, Any]) -> list[dict[str, Any]]:
    s = c["summary"]
    validation = c["validation"].set_index("model")
    factor = c["factor_network"].set_index("sample")
    factor_models = c["factor_models"].set_index("specification")
    vif = c["vif"].pivot(index="component", columns="sample", values="vif")
    episodes = c["episodes"]
    source = c["source_manifest"]
    market = c["market_manifest"]

    coverage_rows = []
    for row in c["coverage"].itertuples():
        coverage_rows.append(
            [
                row.protocol_name,
                row.category,
                str(row.first_observation),
                f"{int(row.valid_daily_returns):,}",
                pct(row.coverage_rate_since_system_start),
            ]
        )

    model_labels = {
        "original_heuristic": "Original heuristic",
        "equal_component_weight": "Equal component weights",
        "stablecoin_only": "Stablecoin contraction",
        "tvl_momentum": "TVL momentum",
        "unconstrained_logistic": "Five-variable logistic",
        "parsimonious_selected": "Chronologically selected",
    }
    performance_rows = []
    for key, label in model_labels.items():
        row = validation.loc[key]
        features = row["selected_features"] if isinstance(row["selected_features"], str) else ""
        performance_rows.append(
            [
                label,
                features or "-",
                fmt(row["average_precision"]),
                fmt(row["roc_auc"]),
                pct(row["precision"]),
                pct(row["recall"]),
            ]
        )

    walk_fold = c["walk"][c["walk"]["fold"] != "aggregate_oof"].copy()
    walk_fold["fold"] = walk_fold["fold"].astype(int)
    walk_pivot = walk_fold.pivot(index="model", columns="fold", values="average_precision")
    walk_agg = c["walk"][c["walk"]["fold"] == "aggregate_oof"].set_index("model")
    walk_rows = []
    for key, label in {
        "original_heuristic": "Original heuristic",
        "stablecoin_only": "Stablecoin contraction",
        "parsimonious_selected": "Selected specification",
    }.items():
        values = [fmt(walk_pivot.loc[key, fold]) for fold in range(1, 5)]
        walk_rows.append([label, *values, fmt(walk_agg.loc[key, "average_precision"])])

    target_pivot = c["target"].pivot(index="target", columns="model", values="average_precision")
    target_prev = c["target"].groupby("target")["holdout_prevalence"].first()
    target_labels = {
        "equal_weight_available": "Equal weight, available",
        "lagged_tvl_weighted": "Lagged-TVL weight",
        "capped_lagged_tvl_weighted": "Capped lagged-TVL",
        "category_balanced": "Category balanced",
        "retrospective_common_history": "Common history (retrospective)",
    }
    target_rows = []
    for key, label in target_labels.items():
        target_rows.append(
            [
                label,
                fmt(target_prev.loc[key]),
                fmt(target_pivot.loc[key, "original_heuristic"]),
                fmt(target_pivot.loc[key, "stablecoin_only"]),
                fmt(target_pivot.loc[key, "parsimonious_selected"]),
            ]
        )

    scenario_hist = c["scenario_historical"].set_index("quantity")["value"]
    scenario_rows = [
        ["Conditional grid p95", pct(scenario_hist["scenario_grid_p95_min"]), pct(scenario_hist["scenario_grid_p95_max"])],
        ["Conditional grid p99", pct(scenario_hist["scenario_grid_p99_min"]), pct(scenario_hist["scenario_grid_p99_max"])],
        ["Historical seven-day loss", pct(scenario_hist["historical_7d_loss_p95"]), pct(scenario_hist["historical_7d_loss_p99"])],
    ]

    protocol_labels = {
        "aave-v3": "Aave V3",
        "compound-v3": "Compound V3",
        "convex-finance": "Convex Finance",
        "curve-dex": "Curve DEX",
        "lido": "Lido",
        "morpho-blue": "Morpho Blue",
        "sky-lending": "Sky Lending",
        "uniswap-v3": "Uniswap V3",
    }
    top_episodes = episodes.sort_values("severity_loss", ascending=False).head(3)
    episode_text = []
    for row in top_episodes.itertuples():
        drivers = []
        for item in row.most_negative_protocols.split("; "):
            slug, value = item.split(":")
            drivers.append(f"{protocol_labels.get(slug, slug)} ({pct(float(value))})")
        episode_text.append(
            f"{row.onset}: {pct(row.severity_loss)} loss, led by " + ", ".join(drivers)
        )

    blocks: list[dict[str, Any]] = [
        section("Abstract"),
        paragraph(
            "Protocol total value locked (TVL) is reported separately even though token prices, collateral, settlement assets, and depositor decisions affect several protocols at once. This paper tests whether aggregate stablecoin contraction, common TVL-return movement, thresholded network density, realised volatility, and downside breadth add out-of-sample information about a decline in an eight-protocol USD TVL basket. USD TVL equals token quantity multiplied by token price, so the outcome combines flows with valuation and accounting effects. Daily DefiLlama observations from 2022-04-16 to 2026-07-10 are evaluated with a purged 70/30 chronological holdout and four expanding walk-forward folds. The seven-day event rate is 0.200 in the final holdout. The prespecified five-component score produces average precision of 0.153, whereas stablecoin contraction alone reaches 0.267. A training-selected inverse-correlation model reaches 0.396, but its selected variables and performance vary across folds and target definitions. ETH adjustment reduces mean holdout correlation from 0.619 to 0.105, indicating that much of the apparent network reflects common market exposure. Lag and episode tests further show that stablecoin information is concentrated close to stress onset. The data support a modest near-term predictive association for aggregate stablecoin contraction, not a causal contagion or dependable early-warning claim. All data-dependent scaling parameters, estimated coefficients, and model operating thresholds were fitted using training observations only. Fixed return bounds, transparent component weights, and display bands were prespecified."
        ),
        section("Introduction"),
        paragraph(
            "Cross-protocol stress is easy to describe after prices and TVL fall together; detecting it beforehand is harder. A correlation edge records simultaneous co-movement, not a transaction, contractual exposure, user overlap, liquidation route, or causal transmission channel. Connectedness statistics in conventional finance often use predictive dependence or forecast-error variance decompositions (Billio et al., 2012; Diebold & Yilmaz, 2012). The graph studied here is deliberately weaker: it is a rolling summary of contemporaneous USD TVL returns. The empirical question is therefore whether those summaries, together with stablecoin supply and broad downside participation, help rank future basket declines."
        ),
        paragraph(
            "The distinction changes how a successful result would be interpreted. Common ETH exposure can make protocol TVL move together without any cross-protocol transmission. Broad withdrawals can create liquidity stress without revealing which protocol caused it. A signal can also measure stress accurately on day <i>t</i> yet contain little information about days <i>t</i>+1 through <i>t</i>+7. The analysis separates common market exposure, simultaneous co-movement, liquidity stress, statistical connectedness, direct contagion, and predictive warning rather than using these labels interchangeably."
        ),
        paragraph(
            f"The main negative result is economically informative. In the final {s['holdout_observations']}-day holdout, adding correlation, density, volatility, and breadth to stablecoin contraction lowers average precision from {fmt(s['stablecoin_holdout_average_precision'])} to {fmt(s['original_holdout_average_precision'])}. Diagnostics trace that loss to common-factor contamination, correlation-density redundancy, sign reversal, and timing. The selected inverse-correlation specification is examined because it was chosen using past data, but its instability prevents it from replacing the original hypothesis with a new universal rule."
        ),
        section("Economic Mechanisms and Testable Hypotheses"),
        paragraph(
            "<b>Stablecoin contraction.</b> Stablecoins provide settlement balances and collateral across DeFi. Redemptions may remove deployable liquidity, and doubts about reserve quality can produce run dynamics or asset sales (Baughman et al., 2022; Ahmed et al., 2024). Aggregate supply contraction could therefore precede lower protocol TVL. The expected sign is positive after expressing the variable as contraction. The same observation can arise from issuer rotation, migration to bank deposits, or a flight to fiat that bypasses the selected protocols. The hypothesis is contradicted if contraction does not outperform prevalence out of sample, loses its ranking ability at operational lags, or is unrelated across basket definitions."
        ),
        paragraph(
            "<b>Common TVL movement and density.</b> Shared collateral repricing, common depositor behaviour, leverage, and liquidations can raise pairwise TVL-return correlations. These channels are consistent with broad stress, but high correlation may also be produced by a common crypto-market beta. Forbes and Rigobon (2002) show why changes in measured correlation need not establish contagion. The expected predictive sign was set positive: tighter co-movement was hypothesised to precede wider declines. Network density counts correlations above 0.45 and is a nonlinear transformation of the same 60-day matrix, so it may add little once average correlation is known. A negative coefficient, strong attenuation after ETH residualisation, or high variance inflation would reject the intended interpretation."
        ),
        paragraph(
            "<b>Volatility and breadth.</b> Realised basket volatility can rise as leverage becomes fragile, but it often reacts after a shock. Downside breadth, measured as the fraction of available protocols with negative daily TVL returns, identifies broad pressure more directly. Both were assigned positive signs. They qualify as warning variables only if their information survives lagging and appears before episode onset. If they peak during or after events, they are coincident stress measures."
        ),
        paragraph(
            "<b>Measurement.</b> For protocol <i>i</i>, USD TVL at time <i>t</i> is the sum of token quantities multiplied by token prices. Its change combines deposits and withdrawals, asset-price movements, token migration, protocol accounting, and source revisions. Schär (2021) and Gudgeon et al. (2020) explain the composability and leverage channels that make DeFi balance sheets interdependent, but the available aggregate endpoint does not identify these channels separately. This paper therefore studies USD TVL stress. It does not estimate depositor flows or direct contagion."
        ),
        section("Data, Measurement, and Sample Construction"),
        paragraph(
            f"The protocol and stablecoin snapshots were retrieved from the exact DefiLlama endpoints on 2026-07-17. The raw protocol file contained {source['rows']['protocol_rows_raw']:,} protocol-date rows; {source['rows']['protocol_date_duplicates_removed']} duplicates were removed by retaining one observation per protocol and UTC date. ETH prices were retrieved from the DefiLlama Coins API on {market['retrieved_utc_date']} and cut off at {market['sample_cutoff']}. The resulting modelling sample begins on {s['sample_start']} after the 60-day network and other rolling features become available and ends on {s['sample_end']} so that every seven-day forward outcome is observable."
        ),
        paragraph(
            "The eight protocols were manually selected to represent lending, decentralised exchanges, liquid staking, collateralised debt, and yield aggregation while retaining sufficiently long endpoint histories. This is not a historical top-eight rule. Compound V3 and Morpho Blue enter later, and the set omits failed, discontinued, and smaller protocols. Selection used end-of-sample knowledge and therefore favours surviving large protocols. The basket is described as a selected cross-protocol sample, never as the DeFi system. A six-protocol common-history sensitivity is reported as retrospective because its membership also uses full-sample availability."
        ),
        table(1, "Protocol sample and observed history", ["Protocol", "Category", "First date", "Valid returns", "Coverage"], coverage_rows, [1.22, 1.05, 1.03, 0.88, 0.78]),
        paragraph(
            "Daily protocol log returns are bounded at -0.35 and 0.35 before aggregation to limit endpoint discontinuities. The bounds are fixed design choices, not fitted winsorisation. The primary target is the equal-weight mean among protocols available on each date; equal weighting emphasises breadth. Alternatives use prior-day TVL weights, capped prior-day TVL weights, category-balanced weights, and a retrospective common-history basket. Lagged weights prevent the target from using contemporaneous size information."
        ),
        section("Empirical Design"),
        paragraph(
            "At the close of day <i>t</i>, the model observes features computed through <i>t</i> and issues a ranking for the cumulative basket return from <i>t</i>+1 through <i>t</i>+7. An event equals one when that return is below -5%. Predictor windows do not include outcome dates. The primary split uses 1,075 training observations through 2025-03-25, a seven-day purge, and 465 holdout observations beginning 2025-04-02. The event prevalence is the no-skill baseline for average precision. Average precision is the stepwise sum of precision gains across recall increments at distinct score thresholds, not a trapezoidal precision-recall area (Saito & Rehmsmeier, 2015)."
        ),
        paragraph(
            "The original score combines training-scaled correlation (0.25), density (0.20), stablecoin contraction (0.20), volatility (0.20), and breadth (0.15). These weights are prespecified heuristics. Comparison models include equal weights, stablecoin contraction, seven-day TVL momentum, an L2 logistic model with all five components, and a parsimonious logistic specification selected only within nested chronological training folds. Operating thresholds maximise training F2 subject to an alert-rate cap; ranking metrics do not depend on those thresholds."
        ),
        paragraph(
            "Four expanding outer folds provide out-of-fold predictions. Each fold estimates scaling, feature selection, coefficients, and thresholds using earlier observations and purges seven days between fit and test periods. Serial dependence is retained with a moving-block bootstrap (Künsch, 1989), while overlapping forward labels are also examined as distinct episodes. This design follows the time-ordering logic of walk-forward and purged validation rather than random resampling (Bergmeir et al., 2018; López de Prado, 2018)."
        ),
        section("Descriptive Evidence"),
        paragraph(
            f"Common ETH exposure explains much of the raw network. Protocol-specific ETH betas are estimated over the prior 180 days, excluding day <i>t</i>, and residual TVL returns subtract the predicted ETH component. In training, mean pairwise correlation falls from {fmt(factor.loc['training', 'raw_mean_correlation'])} to {fmt(factor.loc['training', 'residual_mean_correlation'])}; in the holdout it falls from {fmt(factor.loc['holdout', 'raw_mean_correlation'])} to {fmt(factor.loc['holdout', 'residual_mean_correlation'])}. Holdout density falls from {fmt(factor.loc['holdout', 'raw_mean_density'])} to {fmt(factor.loc['holdout', 'residual_mean_density'])}. The raw graph therefore measures common market exposure to a substantial degree. ETH is only one factor, so the residual is not a pure quantity-flow measure."
        ),
        figure(1, "ETH adjustment sharply attenuates the TVL-return network", "figure_1_eth_factor_adjustment.png"),
        paragraph(
            f"Redundancy is visible before any prediction model is fitted. Correlation and density have a training correlation of {fmt(s['correlation_density_training_correlation'])} and holdout correlation of {fmt(s['correlation_density_holdout_correlation'])}. Their training variance-inflation factors are {fmt(vif.loc['correlation', 'training'], 1)} and {fmt(vif.loc['density', 'training'], 1)}. Stablecoin contraction, volatility, and breadth remain near one. Treating correlation and density as separate 45% of the heuristic score effectively double-counts one matrix."
        ),
        figure(2, "Correlation and density dominate component redundancy", "figure_2_component_correlation.png"),
        paragraph(
            "The component deciles do not support a stable monotone relationship. In the holdout, high correlation and high volatility correspond to lower event rates, opposite to the prespecified signs. Stablecoin contraction is more consistent with the predicted ordering, although its deciles are not perfectly monotone. The distributional shift explains why the composite's lowest score decile is riskier than its highest: several highly weighted components rank in the wrong direction."
        ),
        figure(3, "Holdout event rates reveal weak and reversed component ordering", "figure_3_component_deciles.png"),
        section("Main Out-of-Sample Results"),
        paragraph(
            f"The final holdout contains {s['holdout_events']} events in {s['holdout_observations']} observations, so prevalence is {fmt(s['holdout_prevalence'])}. The original heuristic records average precision of {fmt(s['original_holdout_average_precision'])} and ROC area of {fmt(s['original_holdout_roc_auc'])}; both indicate reversed ranking. Stablecoin contraction is the only prespecified single signal with a clear gain, reaching average precision {fmt(s['stablecoin_holdout_average_precision'])} and ROC area {fmt(s['stablecoin_holdout_roc_auc'])}. The full logistic model remains weak at {fmt(s['full_logistic_holdout_average_precision'])}."
        ),
        paragraph(
            f"Nested training selection chooses correlation alone with a negative coefficient and produces holdout average precision of {fmt(s['parsimonious_holdout_average_precision'])}. That result was not created by reversing the final holdout score; the sign and feature were selected from earlier folds. It is nevertheless a fragile finding. A low-correlation state may indicate protocol-specific withdrawals that reduce the equal-weight basket without creating broad co-movement, or it may reflect a regime-specific measurement pattern. Neither explanation establishes that low connectedness causes stress."
        ),
        table(2, "Final holdout classification and ranking results", ["Model", "Selected feature", "Avg. precision", "ROC area", "Precision", "Recall"], performance_rows, [1.38, 1.12, 0.76, 0.66, 0.67, 0.62]),
        figure(4, "Stablecoin contraction outperforms the original composite", "figure_4_model_performance.png", width=4.45),
        section("Why the Composite Fails"),
        paragraph(
            "The additional variables do not supply independent leading information. Fixed-score ablations show that every stablecoin-plus-one combination underperforms stablecoin alone; the least damaging addition is breadth, at average precision 0.256. In logistic comparisons, adding correlation to stablecoin contraction raises holdout average precision from 0.276 to 0.291 because the fitted correlation coefficient is negative. Adding volatility reduces it to 0.152. The five-variable logistic coefficient on volatility changes from positive in training to negative in a diagnostic holdout fit, while correlation is negative in both periods. These signs contradict the original high-connectedness and high-volatility hypotheses."
        ),
        paragraph(
            f"Price-factor adjustment changes the network but does not rescue the theory. Replacing raw network components with ETH-residual components raises the raw-target composite from {fmt(s['original_holdout_average_precision'])} to {fmt(factor_models.loc['eth_residual_components_raw_target', 'average_precision'])}, approximately the prevalence benchmark. Predicting a residual-TVL target performs much worse. The attenuation shows that common asset-price exposure is central; the residual target also contains more noise and is not equivalent to deposit flow."
        ),
        paragraph(
            f"Scaling saturation is not the explanation. Training 5th-to-95th percentile scaling with clipping produces average precision {fmt(c['scaling'].iloc[0]['average_precision'])}; applying the same linear transform without clipping produces {fmt(c['scaling'].iloc[1]['average_precision'])}. The reversed ranking remains. A more plausible account combines redundant network inputs, incorrect positive signs, common-factor contamination, and predictors that react near or after stress."
        ),
        section("Robustness and Alternative Definitions"),
        paragraph(
            f"Stablecoin contraction is comparatively insensitive to the basket definition: its holdout average precision ranges from {fmt(s['target_stablecoin_ap_min'])} to {fmt(s['target_stablecoin_ap_max'])}. The inverse-correlation model is not. It falls to 0.176 for the lagged-TVL target but exceeds 0.38 for equal-weight, capped, and category-balanced targets. Thus the strongest selected model partly predicts how stress is defined rather than a general cross-protocol state. The common-history result is retrospective and cannot remove survivor bias."
        ),
        table(3, "Sensitivity to the definition of the seven-day stress target", ["Target", "Prevalence", "Original", "Stablecoin", "Selected"], target_rows, [1.80, 0.72, 0.72, 0.76, 0.72]),
        paragraph(
            f"Walk-forward evidence reaches the same narrower conclusion. Across 774 out-of-fold observations, average precision is {fmt(s['walk_forward_original_average_precision'])} for the original score, {fmt(s['walk_forward_stablecoin_average_precision'])} for stablecoin contraction, and {fmt(s['walk_forward_parsimonious_average_precision'])} for the selected specification. The last number conceals substantial dispersion: fold average precision ranges from 0.147 to 0.485, and selection alternates among density, stablecoin plus correlation, and correlation. A specification that changes with each regime is unsuitable as a fixed warning rule."
        ),
        table(4, "Expanding walk-forward average precision", ["Model", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "OOF"], walk_rows, [1.58, 0.62, 0.62, 0.62, 0.62, 0.62]),
        figure(5, "Walk-forward performance varies materially across folds", "figure_5_walk_forward.png", width=4.70),
        paragraph(
            f"Timing tests weaken the early-warning interpretation. For the seven-day target, stablecoin average precision declines from {fmt(s['stablecoin_lag0_h7_ap'])} at day <i>t</i> to {fmt(s['stablecoin_lag7_h7_ap'])} at <i>t</i>-7, close to the 0.200 holdout prevalence. The original score stays near or below prevalence across useful lags. Stablecoin contraction contains near-term information, but little of that ranking advantage remains one week before the outcome window."
        ),
        figure(6, "Stablecoin information decays as the operational lag increases", "figure_6_lag_timing.png"),
        section("Episode-Level Evidence"),
        paragraph(
            f"Overlapping daily labels form {s['episodes']} holdout episodes after consecutive event days are joined. The selected training rule produces a pre-onset alert for {s['episodes_with_pre_onset_alert']} episodes, with a median lead of {fmt(s['median_warning_lead_days'], 0)} day among warned episodes. Two episodes are alerted only after onset, and {s['false_alert_episodes']} alert runs occur outside stress episodes. The number of distinct episodes is much smaller than the {s['holdout_events']} positive daily labels, which is why daily recall alone overstates the effective event count."
        ),
        paragraph(
            "The three most severe episode onsets are " + "; ".join(episode_text) + ". The protocol contributions differ across these dates. That heterogeneity is consistent with basket stress arising from idiosyncratic as well as common movements. Event-time averages show no sustained rise in correlation or volatility far enough before onset to support a reliable warning mechanism."
        ),
        figure(7, "Most warnings arrive close to episode onset and false alerts are frequent", "figure_7_episode_timing.png"),
        section("Conditional Scenario Analysis"),
        paragraph(
            "The simulation is a conditional scenario exercise, not an inferred contagion model. An initial protocol is sampled using current TVL weights; its loss comes from its historical seven-day tail. Losses then pass across correlation edges under varied propagation multipliers, rounds, edge thresholds, tail cutoffs, caps, and network windows. Only the initial loss distribution and current network are empirical inputs. Propagation strength, Uniform(0.75, 1.25) dispersion, round count, edge cutoff, and cap are heuristic."
        ),
        paragraph(
            f"Across the structured grid, p95 basket loss ranges from {pct(s['scenario_p95_min'])} to {pct(s['scenario_p95_max'])}, and p99 ranges from {pct(s['scenario_p99_min'])} to {pct(s['scenario_p99_max'])}. Historical seven-day p99 loss is {pct(s['historical_7d_loss_p99'])}, with a maximum of {pct(scenario_hist['historical_7d_loss_max'])}. Path-count convergence addresses Monte Carlo sampling error conditional on a parameter set; the wide grid addresses parameter uncertainty, and neither resolves structural uncertainty about whether correlation represents a propagation channel."
        ),
        table(5, "Conditional scenario ranges and historical comparison", ["Quantity", "Lower", "Upper"], scenario_rows, [2.40, 1.05, 1.05]),
        figure(8, "Tail-loss estimates depend strongly on heuristic scenario parameters", "figure_8_scenario_sensitivity.png"),
        section("Discussion"),
        paragraph(
            "Aggregate stablecoin contraction is the only prespecified signal that repeatedly improves ranking over prevalence, and even that advantage is concentrated near stress onset. A plausible mechanism is withdrawal of settlement liquidity before basket declines. Aggregate issuance changes also reflect redemption, issuer substitution, and movement outside the selected protocols, so the coefficient cannot identify the mechanism. The result supports monitoring stablecoin contraction alongside other information, not treating it as an autonomous trading signal."
        ),
        paragraph(
            "High TVL-return correlation does not behave as hypothesised. ETH adjustment removes most of the raw network, and the remaining predictive sign is often negative. The correlation graph is still a useful description of simultaneous common exposure. It does not measure direct economic links and does not provide stable advance information about the selected basket. A stronger contagion design would require user-overlap, collateral-flow, liquidation, bridge, or contractual exposure data with timestamps that permit transmission to be ordered."
        ),
        section("Limitations"),
        paragraph(
            "Four constraints remain. First, the manually selected survivor sample does not represent the historical DeFi universe. Second, USD TVL mixes price and quantity; one ETH factor cannot recover token flows or remove all common crypto exposures. Third, overlapping labels, source revisions, and only 26 holdout episodes limit effective sample size. Fourth, the final holdout was kept out of fitting, but its diagnostics are now part of this report; there is no second untouched period in which to confirm the inverse-correlation anomaly. The analysis is predictive and associational, not causal."
        ),
        section("Conclusion"),
        paragraph(
            "The original hypothesis predicted that stablecoin contraction, high common TVL movement, dense correlations, volatility, and downside breadth would jointly precede a seven-day basket decline. They do not. The composite ranks holdout outcomes worse than prevalence, mainly because its network variables are redundant, heavily exposed to ETH, and signed incorrectly in the later regime. Stablecoin contraction retains a modest near-term association across target definitions, but its advantage nearly disappears at a seven-day predictor lag. The conclusion that survives is narrow: aggregate stablecoin supply contains some short-horizon information about future USD TVL stress in this selected basket. Direct contagion and dependable early warning remain untested."
        ),
        section("References"),
        reference("Ahmed, R., Aldasoro, I., & Duley, C. (2024). Public information and stablecoin runs (BIS Working Papers No. 1164). Bank for International Settlements. https://www.bis.org/publ/work1164.htm"),
        reference("Bartoletti, M., Chiang, J. H.-Y., & Lluch Lafuente, A. (2021). SoK: Lending pools in decentralized finance. In M. Bernhard, A. Bracciali, L. Gudgeon, T. Haines, A. Klages-Mundt, S. Matsuo, D. Perez, M. Sala, & S. Werner (Eds.), Financial cryptography and data security: FC 2021 International Workshops (pp. 553-578). Springer. https://doi.org/10.1007/978-3-662-63958-0_40"),
        reference("Baughman, G., Carapella, F., Gerszten, J., & Mills, D. (2022, December 16). The stable in stablecoins. FEDS Notes. Board of Governors of the Federal Reserve System. https://doi.org/10.17016/2380-7172.3220"),
        reference("Bergmeir, C., Hyndman, R. J., & Koo, B. (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. Computational Statistics & Data Analysis, 120, 70-83. https://doi.org/10.1016/j.csda.2017.11.003"),
        reference("Billio, M., Getmansky, M., Lo, A. W., & Pelizzon, L. (2012). Econometric measures of connectedness and systemic risk in the finance and insurance sectors. Journal of Financial Economics, 104(3), 535-559. https://doi.org/10.1016/j.jfineco.2011.12.010"),
        reference("DefiLlama. (2026a). Protocol TVL endpoint [Data set]. Retrieved July 17, 2026, from https://api.llama.fi/protocol/{slug}"),
        reference("DefiLlama. (2026b). Stablecoin supply endpoint [Data set]. Retrieved July 17, 2026, from https://stablecoins.llama.fi/stablecoincharts/all"),
        reference("DefiLlama. (2026c). Ethereum daily price endpoint [Data set]. Retrieved July 20, 2026, from https://coins.llama.fi/chart/coingecko:ethereum"),
        reference("Diebold, F. X., & Yilmaz, K. (2012). Better to give than to receive: Predictive directional measurement of volatility spillovers. International Journal of Forecasting, 28(1), 57-66. https://doi.org/10.1016/j.ijforecast.2011.02.006"),
        reference("Forbes, K. J., & Rigobon, R. (2002). No contagion, only interdependence: Measuring stock market comovements. The Journal of Finance, 57(5), 2223-2261. https://doi.org/10.1111/0022-1082.00494"),
        reference("Gudgeon, L., Perez, D., Harz, D., Livshits, B., & Gervais, A. (2020). The decentralized financial crisis. 2020 Crypto Valley Conference on Blockchain Technology, 1-15. https://doi.org/10.1109/CVCBT50464.2020.00005"),
        reference("Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. The Annals of Statistics, 17(3), 1217-1241. https://doi.org/10.1214/aos/1176347265"),
        reference("López de Prado, M. (2018). Advances in financial machine learning. Wiley."),
        reference("Qin, K., Zhou, L., Livshits, B., & Gervais, A. (2021). Attacking the DeFi ecosystem with flash loans for fun and profit. In N. Borisov & C. Diaz (Eds.), Financial cryptography and data security (pp. 3-32). Springer. https://doi.org/10.1007/978-3-662-64322-8_1"),
        reference("Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432"),
        reference("Schär, F. (2021). Decentralized finance: On blockchain- and smart contract-based financial markets. Federal Reserve Bank of St. Louis Review, 103(2), 153-174. https://doi.org/10.20955/r.103.153-74"),
        reference("Werner, S. M., Perez, D., Gudgeon, L., Klages-Mundt, A., Harz, D., & Knottenbelt, W. J. (2022). SoK: Decentralized finance (DeFi). Proceedings of the 4th ACM Conference on Advances in Financial Technologies, 30-46. https://doi.org/10.1145/3558535.3559780"),
        section("Reproducibility Appendix"),
        paragraph(
            "The repository contains the immutable raw snapshots, extraction metadata, feature code, model outputs, figures, and report source. From the repository root, install the pinned Python requirements and run the following command. Offline mode preserves the snapshot timestamps and does not request newer observations. The pipeline rebuilds staged and processed data, market controls, model outputs, figures, this report, and validation checks."
        ),
        code("USE_EXISTING_RAW=1 ./run_pipeline.sh"),
        paragraph(
            "Repository: <link href='" + REPOSITORY + "' color='#1F5A7A'>" + REPOSITORY + "</link>"
        ),
    ]
    return blocks


def register_fonts() -> None:
    root = Path("/System/Library/Fonts/Supplemental")
    pdfmetrics.registerFont(TTFont("TNR", root / "Times New Roman.ttf"))
    pdfmetrics.registerFont(TTFont("TNR-Bold", root / "Times New Roman Bold.ttf"))
    pdfmetrics.registerFont(TTFont("TNR-Italic", root / "Times New Roman Italic.ttf"))
    pdfmetrics.registerFont(TTFont("TNR-BoldItalic", root / "Times New Roman Bold Italic.ttf"))
    pdfmetrics.registerFontFamily(
        "TNR",
        normal="TNR",
        bold="TNR-Bold",
        italic="TNR-Italic",
        boldItalic="TNR-BoldItalic",
    )


def styles() -> dict[str, ParagraphStyle]:
    return {
        "title": ParagraphStyle("title", fontName="TNR-Bold", fontSize=18, leading=20, alignment=TA_CENTER, spaceAfter=3),
        "subtitle": ParagraphStyle("subtitle", fontName="TNR-Italic", fontSize=12, leading=14, alignment=TA_CENTER, spaceAfter=9),
        "meta": ParagraphStyle("meta", fontName="TNR", fontSize=9.3, leading=11.2, alignment=TA_CENTER, spaceAfter=1),
        "section": ParagraphStyle("section", fontName="TNR-Bold", fontSize=12.2, leading=14, alignment=TA_LEFT, spaceBefore=7, spaceAfter=3, keepWithNext=True),
        "body": ParagraphStyle("body", fontName="TNR", fontSize=9.15, leading=11.15, alignment=TA_JUSTIFY, firstLineIndent=0.20 * inch, spaceAfter=4.2, allowWidows=0, allowOrphans=0),
        "caption_label": ParagraphStyle("caption_label", fontName="TNR-Bold", fontSize=8.4, leading=9.6, alignment=TA_CENTER, spaceBefore=2, spaceAfter=0),
        "caption": ParagraphStyle("caption", fontName="TNR-Italic", fontSize=8.4, leading=9.6, alignment=TA_CENTER, spaceAfter=4),
        "reference": ParagraphStyle("reference", fontName="TNR", fontSize=7.15, leading=8.25, alignment=TA_LEFT, leftIndent=0.20 * inch, firstLineIndent=-0.20 * inch, spaceAfter=1.8),
        "table": ParagraphStyle("table", fontName="TNR", fontSize=7.6, leading=8.7, alignment=TA_LEFT),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=8.2, leading=10, alignment=TA_CENTER, backColor=colors.HexColor("#F1F3F4"), borderPadding=5, spaceBefore=2, spaceAfter=5),
    }


def page_number(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setFont("TNR", 8)
    canvas.setFillColor(colors.HexColor("#333333"))
    canvas.drawCentredString(letter[0] / 2, 0.30 * inch, str(doc.page))
    canvas.restoreState()


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def build_pdf(blocks: list[dict[str, Any]]) -> Path:
    register_fonts()
    st = styles()
    output = REPORT_DIR / "technical_report.pdf"
    doc = SimpleDocTemplate(
        str(output),
        pagesize=letter,
        leftMargin=0.62 * inch,
        rightMargin=0.62 * inch,
        topMargin=0.52 * inch,
        bottomMargin=0.52 * inch,
        title=f"{TITLE}: {SUBTITLE}",
        author=AUTHOR,
        subject=COURSE,
    )
    story: list[Any] = [
        Paragraph(TITLE, st["title"]),
        Paragraph(SUBTITLE, st["subtitle"]),
        Paragraph(AUTHOR, st["meta"]),
        Paragraph(COURSE, st["meta"]),
        Paragraph(DATE, st["meta"]),
        Paragraph(f"<link href='{REPOSITORY}' color='#1F5A7A'>{REPOSITORY}</link>", st["meta"]),
        Spacer(1, 5),
    ]

    for block in blocks:
        kind = block["kind"]
        if kind == "section":
            story.append(Paragraph(safe_text(block["title"]), st["section"]))
        elif kind == "paragraph":
            story.append(Paragraph(block["text"], st["body"]))
        elif kind == "reference":
            story.append(Paragraph(safe_text(block["text"]), st["reference"]))
        elif kind == "code":
            story.append(Paragraph(safe_text(block["text"]), st["code"]))
        elif kind == "figure":
            path = FIGURES_DIR / block["filename"]
            width = block["width"] * inch
            height = width * 800 / 1500
            img = Image(str(path), width=width, height=height)
            img.hAlign = "CENTER"
            story.append(
                KeepTogether(
                    [
                        img,
                        Paragraph(f"Figure {block['number']}", st["caption_label"]),
                        Paragraph(safe_text(block["title"]), st["caption"]),
                    ]
                )
            )
        elif kind == "table":
            data = [[Paragraph(f"<b>{safe_text(v)}</b>", st["table"]) for v in block["headers"]]]
            data.extend([[Paragraph(safe_text(v), st["table"]) for v in row] for row in block["rows"]])
            tbl = Table(data, colWidths=[w * inch for w in block["widths"]], repeatRows=1, hAlign="CENTER")
            tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7ECEF")),
                        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#111111")),
                        ("FONTNAME", (0, 0), (-1, 0), "TNR-Bold"),
                        ("FONTNAME", (0, 1), (-1, -1), "TNR"),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#5B6770")),
                        ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.HexColor("#5B6770")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F8F9")]),
                        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.append(
                KeepTogether(
                    [
                        tbl,
                        Paragraph(f"Table {block['number']}", st["caption_label"]),
                        Paragraph(safe_text(block["title"]), st["caption"]),
                    ]
                )
            )

    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    return output


def html_to_markdown(text: str) -> str:
    text = re.sub(r"<link href='([^']+)'[^>]*>(.*?)</link>", r"[\2](\1)", text)
    text = text.replace("<i>", "*").replace("</i>", "*")
    text = text.replace("<b>", "**").replace("</b>", "**")
    return html.unescape(re.sub(r"<[^>]+>", "", text))


def build_markdown(blocks: list[dict[str, Any]]) -> Path:
    lines = [
        f"# {TITLE}",
        f"*{SUBTITLE}*",
        "",
        AUTHOR,
        COURSE,
        DATE,
        REPOSITORY,
        "",
    ]
    for block in blocks:
        kind = block["kind"]
        if kind == "section":
            lines.extend([f"## {block['title']}", ""])
        elif kind in {"paragraph", "reference"}:
            lines.extend([html_to_markdown(block["text"]), ""])
        elif kind == "code":
            lines.extend(["```bash", block["text"], "```", ""])
        elif kind == "figure":
            lines.extend(
                [
                    f"<p align=\"center\"><img src=\"../figures/{block['filename']}\" width=\"760\"></p>",
                    f"<p align=\"center\"><strong>Figure {block['number']}</strong><br><em>{block['title']}</em></p>",
                    "",
                ]
            )
        elif kind == "table":
            lines.append("| " + " | ".join(block["headers"]) + " |")
            lines.append("| " + " | ".join(["---"] * len(block["headers"])) + " |")
            for row in block["rows"]:
                lines.append("| " + " | ".join(str(v) for v in row) + " |")
            lines.extend(
                [
                    f"<p align=\"center\"><strong>Table {block['number']}</strong><br><em>{block['title']}</em></p>",
                    "",
                ]
            )
    output = REPORT_DIR / "technical_report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    ensure_directories()
    context = load_context()
    blocks = build_blocks(context)
    markdown = build_markdown(blocks)
    pdf = build_pdf(blocks)
    print(f"Wrote {markdown}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
