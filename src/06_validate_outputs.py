from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from pypdf import PdfReader

from config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, REPORT_DIR, STAGING_DIR


REQUIRED_SECTIONS = [
    "Abstract",
    "Introduction",
    "Economic Mechanisms and Testable Hypotheses",
    "Data, Measurement, and Sample Construction",
    "Empirical Design",
    "Descriptive Evidence",
    "Main Out-of-Sample Results",
    "Why the Composite Fails",
    "Robustness and Alternative Definitions",
    "Episode-Level Evidence",
    "Conditional Scenario Analysis",
    "Discussion",
    "Limitations",
    "Conclusion",
    "References",
    "Reproducibility Appendix",
]

REQUIRED_FIGURES = [f"figure_{number}_{name}.png" for number, name in [
    (1, "eth_factor_adjustment"),
    (2, "component_correlation"),
    (3, "component_deciles"),
    (4, "model_performance"),
    (5, "walk_forward"),
    (6, "lag_timing"),
    (7, "episode_timing"),
    (8, "scenario_sensitivity"),
]]

def font_names(reader: PdfReader) -> set[str]:
    names: set[str] = set()
    for page in reader.pages:
        resources = page.get("/Resources")
        if resources is None:
            continue
        fonts = resources.get_object().get("/Font")
        if fonts is None:
            continue
        for ref in fonts.get_object().values():
            names.add(str(ref.get_object().get("/BaseFont", "")))
    return names


def main() -> None:
    markdown_path = REPORT_DIR / "technical_report.md"
    pdf_path = REPORT_DIR / "technical_report.pdf"
    markdown = markdown_path.read_text(encoding="utf-8")
    reader = PdfReader(str(pdf_path))
    pdf_pages = [(page.extract_text() or "").strip() for page in reader.pages]
    pdf_text = "\n".join(pdf_pages)

    with open(MODELS_DIR / "model_summary.json", encoding="utf-8") as handle:
        summary = json.load(handle)
    with open(STAGING_DIR / "source_manifest.json", encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    with open(PROCESSED_DIR.parent / "raw" / "market_manifest.json", encoding="utf-8") as handle:
        market_manifest = json.load(handle)

    validation = pd.read_csv(MODELS_DIR / "validation_metrics.csv").set_index("model")
    walk = pd.read_csv(MODELS_DIR / "walk_forward_metrics.csv")
    walk_predictions = pd.read_csv(MODELS_DIR / "walk_forward_predictions.csv")
    targets = pd.read_csv(MODELS_DIR / "target_definitions_daily.csv")
    risk_scores = pd.read_csv(MODELS_DIR / "risk_scores.csv")
    processed = pd.read_csv(PROCESSED_DIR / "protocol_returns.csv")
    staging = pd.read_csv(STAGING_DIR / "protocol_tvl.csv")
    market = pd.read_csv(PROCESSED_DIR / "eth_market_control.csv", parse_dates=["date"])
    scenario = pd.read_csv(MODELS_DIR / "scenario_parameter_sensitivity.csv")
    target_sensitivity = pd.read_csv(MODELS_DIR / "target_definition_sensitivity.csv")
    source_code = (Path(__file__).parent / "model_liquidity_stress.py").read_text(encoding="utf-8")

    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, passed: bool, detail: str) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    staging_duplicates = int(staging.duplicated(["date", "protocol_slug"]).sum())
    processed_duplicates = int(processed.duplicated(["date", "protocol_slug"]).sum())
    record("unique_protocol_dates", staging_duplicates == 0 and processed_duplicates == 0,
           f"staging duplicates={staging_duplicates}; processed duplicates={processed_duplicates}")

    label_cols = [column for column in targets if column.startswith("label_7d_")]
    unavailable_target_labels = int(targets[label_cols].tail(7).notna().sum().sum())
    unavailable_primary_labels = int(risk_scores["forward_7d_stress_event"].tail(7).notna().sum())
    record(
        "unavailable_forward_labels_preserved",
        unavailable_target_labels == 0 and unavailable_primary_labels == 0,
        f"non-missing unavailable labels={unavailable_target_labels + unavailable_primary_labels}",
    )

    outer = walk[walk["fold"].astype(str) != "aggregate_oof"].copy()
    outer["train_end"] = pd.to_datetime(outer["train_end"])
    outer["preprocessing_fit_end"] = pd.to_datetime(outer["preprocessing_fit_end"])
    outer["test_start"] = pd.to_datetime(outer["test_start"])
    chronology_ok = bool(
        (outer["preprocessing_fit_end"] <= outer["train_end"]).all()
        and (outer["train_end"] < outer["test_start"]).all()
        and (outer["purge_days"] == 7).all()
    )
    record(
        "walk_forward_fold_local_timing",
        chronology_ok,
        f"outer rows={len(outer)}; minimum gap={(outer['test_start'] - outer['train_end']).dt.days.min()} days",
    )

    oof_duplicates = int(walk_predictions.duplicated(["date", "model"]).sum())
    record("walk_forward_oof_unique", oof_duplicates == 0, f"duplicate model-date predictions={oof_duplicates}")

    lagged_code_ok = "lagged_tvl = tvl.shift(1)" in source_code and "cap_weights(lagged_tvl.loc[date])" in source_code
    target_names = set(target_sensitivity["target"])
    alternative_targets_ok = {
        "equal_weight_available",
        "lagged_tvl_weighted",
        "capped_lagged_tvl_weighted",
        "category_balanced",
        "retrospective_common_history",
    }.issubset(target_names)
    record(
        "lagged_weights_and_alternative_targets",
        lagged_code_ok and alternative_targets_ok,
        f"targets={sorted(target_names)}",
    )

    market_dates = market["date"].sort_values()
    market_complete = bool(
        market_dates.iloc[0].strftime("%Y-%m-%d") == "2021-01-01"
        and market_dates.iloc[-1].strftime("%Y-%m-%d") == market_manifest["sample_cutoff"]
        and market_dates.diff().dropna().dt.days.eq(1).all()
    )
    record("market_control_calendar_complete", market_complete, f"observations={len(market_dates)}")

    model_pairs = {
        "original_holdout_average_precision": ("original_heuristic", "average_precision"),
        "original_holdout_roc_auc": ("original_heuristic", "roc_auc"),
        "stablecoin_holdout_average_precision": ("stablecoin_only", "average_precision"),
        "stablecoin_holdout_roc_auc": ("stablecoin_only", "roc_auc"),
        "full_logistic_holdout_average_precision": ("unconstrained_logistic", "average_precision"),
        "parsimonious_holdout_average_precision": ("parsimonious_selected", "average_precision"),
    }
    differences = {
        key: abs(float(summary[key]) - float(validation.loc[model, metric]))
        for key, (model, metric) in model_pairs.items()
    }
    record("model_summary_reconciles", max(differences.values()) < 1e-12, f"maximum difference={max(differences.values()):.2e}")

    required_report_values = [
        f"{summary['original_holdout_average_precision']:.3f}",
        f"{summary['stablecoin_holdout_average_precision']:.3f}",
        f"{summary['parsimonious_holdout_average_precision']:.3f}",
        f"{summary['raw_holdout_mean_correlation']:.3f}",
        f"{summary['residual_holdout_mean_correlation']:.3f}",
        f"{summary['walk_forward_original_average_precision']:.3f}",
        f"{summary['walk_forward_stablecoin_average_precision']:.3f}",
        f"{summary['stablecoin_lag7_h7_ap']:.3f}",
        str(summary["episodes"]),
        str(summary["false_alert_episodes"]),
    ]
    absent_values = [value for value in required_report_values if value not in markdown]
    record("report_values_match_outputs", not absent_values, f"absent values={absent_values}")

    required_sentence = (
        "All data-dependent scaling parameters, estimated coefficients, and model operating thresholds were fitted "
        "using training observations only. Fixed return bounds, transparent component weights, and display bands "
        "were prespecified."
    )
    record("fitting_description_accurate", required_sentence in markdown, "required calibration sentence present")

    terminology_ok = (
        "PR-AUC" not in markdown
        and "leakage-free" not in markdown.lower()
        and "DeFi Contagion Early-Warning" not in markdown
        and all(phrase not in markdown.lower() for phrase in ["detect contagion", "contagion score", "measures contagion"])
    )
    record("terminology_boundaries", terminology_ok, "average precision and causal boundaries checked")

    scenario_grid = scenario[scenario["scenario"].str.startswith("grid_")]
    scenario_assumptions_present = all(
        term in markdown
        for term in ["propagation multipliers", "rounds", "edge thresholds", "tail cutoffs", "caps", "network windows"]
    )
    record(
        "scenario_grid_and_assumptions",
        len(scenario_grid) == 64 and scenario_assumptions_present,
        f"grid rows={len(scenario_grid)}",
    )

    missing_sections = [name for name in REQUIRED_SECTIONS if f"## {name}" not in markdown]
    numbered_sections = re.findall(r"^##\s+\d", markdown, flags=re.MULTILINE)
    record("academic_structure", not missing_sections and not numbered_sections,
           f"missing sections={missing_sections}; numbered headings={len(numbered_sections)}")

    missing_figures = [name for name in REQUIRED_FIGURES if not (FIGURES_DIR / name).exists()]
    captions = re.findall(r"<strong>Figure (\d+)</strong>", markdown)
    table_captions = re.findall(r"<strong>Table (\d+)</strong>", markdown)
    record(
        "figures_tables_and_captions",
        not missing_figures and captions == [str(i) for i in range(1, 9)] and table_captions == [str(i) for i in range(1, 6)],
        f"missing figures={missing_figures}; figure captions={captions}; table captions={table_captions}",
    )

    corrected_records = [
        "10.1007/978-3-662-63958-0_40",
        "10.1007/978-3-662-64322-8_1",
        "10.1016/j.jfineco.2011.12.010",
        "10.1016/j.ijforecast.2011.02.006",
        "10.1109/CVCBT50464.2020.00005",
        "10.20955/r.103.153-74",
        "10.1145/3558535.3559780",
    ]
    references_ok = all(record_value in markdown for record_value in corrected_records) and "Schär, F." in markdown
    record("verified_reference_records", references_ok, f"required DOI records={len(corrected_records)}")

    endpoint_ok = (
        source_manifest["sources"]["protocol"] in markdown
        and source_manifest["sources"]["stablecoin_supply"] in markdown
        and "https://coins.llama.fi/chart/coingecko:ethereum" in markdown
    )
    record("exact_data_endpoints", endpoint_ok, "protocol, stablecoin, and ETH endpoints present")

    placeholders = re.findall(r"\[(?:Author Name|Course Title)\]|\b(?:TBD|TODO|Lorem ipsum)\b", markdown, re.I)
    record("no_placeholders", not placeholders, f"placeholders={placeholders}")

    identity_ok = (
        str(reader.metadata.author or "") == "Yifan Zhi"
        and "Yifan Zhi" in pdf_text
        and "STAT8308 Blockchain data analytics [Section SA, 2025]" in pdf_text
    )
    record("report_identity", identity_ok, f"PDF author={reader.metadata.author}")

    legacy_text_ok = (
        "DeFi Liquidity Stress Monitoring Prototype" not in pdf_text
        and "Keywords:" not in pdf_text
        and "Keywords:" not in markdown
        and not re.search(r"^Note\.", markdown, flags=re.MULTILINE)
    )
    record("legacy_footer_keywords_notes_removed", legacy_text_ok, "legacy footer, keywords, and notes absent")

    page_count = len(reader.pages)
    short_pages = [number for number, text in enumerate(pdf_pages, start=1) if len(text) < 500]
    record("pdf_page_count_and_density", 5 <= page_count <= 10 and not short_pages,
           f"pages={page_count}; near-empty pages={short_pages}")

    embedded = font_names(reader)
    times = sorted(name for name in embedded if "TimesNewRoman" in name.replace(" ", ""))
    record("times_new_roman_embedded", bool(times), f"matching fonts={times}")

    repository_url = "https://github.com/Zzhi993/defi-liquidity-stress-monitor"
    record("repository_link_present", repository_url in markdown and repository_url in pdf_text, repository_url)

    project_root = Path(__file__).resolve().parent.parent
    leaked_review_files = []
    for path in project_root.rglob("*"):
        relative = path.relative_to(project_root)
        internal_directory = any(part.lower() in {"review_only", "internal_review"} for part in relative.parts)
        internal_filename = "audit" in path.name.lower() or path.name.lower().startswith("review_")
        if internal_directory or internal_filename:
            leaked_review_files.append(str(relative))
    record("internal_review_material_excluded", not leaked_review_files, f"leaked files={leaked_review_files}")

    passed = all(result["passed"] for result in checks.values())
    receipt = {
        "status": "pass" if passed else "fail",
        "data_snapshot_utc": source_manifest["snapshot_utc"],
        "market_control_retrieved": market_manifest["retrieved_utc_date"],
        "report_pages": page_count,
        "checks": checks,
    }
    output = MODELS_DIR / "quality_control.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")

    if not passed:
        failed = [name for name, result in checks.items() if not result["passed"]]
        raise RuntimeError(f"Output validation failed: {', '.join(failed)}")
    print(f"Validated {len(checks)} checks; receipt written to {output}.")


if __name__ == "__main__":
    main()
