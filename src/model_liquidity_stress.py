from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    BLOCK_BOOTSTRAP_LENGTH,
    BLOCK_BOOTSTRAP_REPLICATIONS,
    EDGE_CORRELATION_THRESHOLD,
    FORECAST_HORIZON_DAYS,
    MAX_TRAINING_ALERT_RATE,
    MIN_ACTIVE_PROTOCOLS,
    MIN_PAIR_OBSERVATIONS,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    ROLLING_WINDOW_DAYS,
    TRAIN_SHARE,
    ensure_directories,
)


COMPONENTS = ["correlation", "density", "stablecoin_contraction", "volatility", "breadth"]
HEURISTIC_WEIGHTS = {
    "correlation": 0.25,
    "density": 0.20,
    "stablecoin_contraction": 0.20,
    "volatility": 0.20,
    "breadth": 0.15,
}
TARGET_NAMES = [
    "equal_weight_available",
    "lagged_tvl_weighted",
    "capped_lagged_tvl_weighted",
    "category_balanced",
    "retrospective_common_history",
]


@dataclass
class Split:
    train_mask: pd.Series
    test_mask: pd.Series
    train_end: pd.Timestamp
    holdout_start: pd.Timestamp


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def fit_logistic(
    x: pd.DataFrame, y: pd.Series, l2: float = 0.10, max_iter: int = 200
) -> tuple[float, pd.Series, pd.Series, pd.Series]:
    means = x.mean()
    stds = x.std(ddof=0).replace(0.0, 1.0)
    z = ((x - means) / stds).to_numpy(dtype=float)
    truth = y.to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(z)), z])
    prevalence = float(np.clip(truth.mean(), 1e-5, 1 - 1e-5))
    coefficients = np.zeros(design.shape[1])
    coefficients[0] = math.log(prevalence / (1.0 - prevalence))
    penalty = np.eye(design.shape[1]) * l2
    penalty[0, 0] = 0.0

    for _ in range(max_iter):
        probabilities = sigmoid(design @ coefficients)
        weights = np.clip(probabilities * (1.0 - probabilities), 1e-6, None)
        gradient = design.T @ (probabilities - truth) / len(truth) + penalty @ coefficients
        hessian = (design.T * weights) @ design / len(truth) + penalty
        step = np.linalg.solve(hessian + np.eye(len(coefficients)) * 1e-10, gradient)
        coefficients -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(coefficients[0]), pd.Series(coefficients[1:], index=x.columns), means, stds


def logistic_predict(
    x: pd.DataFrame, intercept: float, coefficients: pd.Series, means: pd.Series, stds: pd.Series
) -> pd.Series:
    standardized = (x[coefficients.index] - means[coefficients.index]) / stds[coefficients.index]
    return pd.Series(sigmoid(intercept + standardized.to_numpy() @ coefficients.to_numpy()), index=x.index)


def forward_sum(series: pd.Series, horizon: int) -> pd.Series:
    future = pd.concat([series.shift(-offset) for offset in range(1, horizon + 1)], axis=1)
    return future.sum(axis=1, min_count=horizon)


def stress_label(forward_return: pd.Series, drawdown: float = 0.05) -> pd.Series:
    output = pd.Series(np.nan, index=forward_return.index, dtype=float)
    available = forward_return.notna()
    output.loc[available] = (forward_return.loc[available] <= math.log(1.0 - drawdown)).astype(int)
    return output


def average_precision(y_true: pd.Series, scores: pd.Series) -> float:
    frame = pd.DataFrame({"y": y_true.astype(int), "score": scores.astype(float)}).sort_values(
        "score", ascending=False, kind="mergesort"
    )
    positives = int(frame["y"].sum())
    if positives == 0:
        return float("nan")
    frame["tp"] = frame["y"].cumsum()
    frame["fp"] = (1 - frame["y"]).cumsum()
    points = frame.loc[frame["score"].ne(frame["score"].shift(-1)), ["tp", "fp"]].copy()
    points["precision"] = points["tp"] / (points["tp"] + points["fp"])
    points["recall"] = points["tp"] / positives
    return float((points["recall"].diff().fillna(points["recall"]) * points["precision"]).sum())


def roc_auc(y_true: pd.Series, scores: pd.Series) -> float:
    truth = y_true.astype(int).reset_index(drop=True)
    values = scores.astype(float).reset_index(drop=True)
    positives = int(truth.sum())
    negatives = int(len(truth) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = values.rank(method="average")
    rank_sum = float(ranks[truth == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def confusion(y_true: pd.Series, predictions: pd.Series) -> tuple[int, int, int, int]:
    y = y_true.astype(int).to_numpy()
    p = predictions.astype(int).to_numpy()
    return (
        int(((y == 1) & (p == 1)).sum()),
        int(((y == 0) & (p == 1)).sum()),
        int(((y == 1) & (p == 0)).sum()),
        int(((y == 0) & (p == 0)).sum()),
    )


def choose_threshold(y_true: pd.Series, scores: pd.Series) -> float:
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 301)))
    best = (-1.0, -1.0, -1.0, float(candidates[-1]))
    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        if float(predictions.mean()) > MAX_TRAINING_ALERT_RATE:
            continue
        tp, fp, fn, _ = confusion(y_true, predictions)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f2 = 5.0 * precision * recall / (4.0 * precision + recall) if 4.0 * precision + recall else 0.0
        candidate = (f2, recall, precision, float(threshold))
        if candidate > best:
            best = candidate
    return float(best[-1])


def metric_row(y_true: pd.Series, scores: pd.Series, threshold: float) -> dict[str, float | int]:
    predictions = (scores >= threshold).astype(int)
    tp, fp, fn, tn = confusion(y_true, predictions)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5.0 * precision * recall / (4.0 * precision + recall) if 4.0 * precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "observations": len(y_true),
        "events": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "threshold": float(threshold),
        "alerts": int(predictions.sum()),
        "alert_rate": float(predictions.mean()),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "roc_auc": roc_auc(y_true, scores),
        "average_precision": average_precision(y_true, scores),
        "brier_score": float(np.mean((scores.clip(0.0, 1.0) - y_true) ** 2)),
    }


def make_split(frame: pd.DataFrame, horizon: int = 7) -> Split:
    split_index = min(max(int(len(frame) * TRAIN_SHARE), 1), len(frame) - 1)
    holdout_start = pd.Timestamp(frame.iloc[split_index]["date"])
    purge_boundary = holdout_start - pd.Timedelta(days=horizon)
    train_mask = frame["date"] < purge_boundary
    test_mask = frame["date"] >= holdout_start
    return Split(train_mask, test_mask, pd.Timestamp(frame.loc[train_mask, "date"].max()), holdout_start)


def build_network(returns: pd.DataFrame, window: int, threshold: float) -> pd.DataFrame:
    min_obs = max(20, math.ceil(window * 0.75))
    rows: list[dict] = []
    for end in range(window - 1, len(returns)):
        sample = returns.iloc[end - window + 1 : end + 1]
        active = [column for column in sample if int(sample[column].notna().sum()) >= min_obs]
        correlations: list[float] = []
        if len(active) >= 2:
            matrix = sample[active].corr(min_periods=min_obs)
            for i, source in enumerate(active):
                for target in active[i + 1 :]:
                    value = matrix.loc[source, target]
                    if pd.notna(value):
                        correlations.append(float(value))
        rows.append(
            {
                "date": returns.index[end],
                "correlation": float(np.mean(correlations)) if correlations else np.nan,
                "density": float(np.mean(np.asarray(correlations) >= threshold)) if correlations else np.nan,
                "network_protocols": len(active),
                "network_pairs": len(correlations),
            }
        )
    return pd.DataFrame(rows).set_index("date")


def cap_weights(values: pd.Series, cap: float = 0.25) -> pd.Series:
    raw = values.where(values > 0).dropna().astype(float)
    if raw.empty:
        return pd.Series(np.nan, index=values.index)
    weights = raw / raw.sum()
    fixed = pd.Series(False, index=weights.index)
    for _ in range(len(weights) + 1):
        over = (weights > cap + 1e-12) & ~fixed
        if not over.any():
            break
        fixed |= over
        weights.loc[fixed] = cap
        free = ~fixed
        remaining = 1.0 - float(weights.loc[fixed].sum())
        if free.any():
            weights.loc[free] = remaining * raw.loc[free] / raw.loc[free].sum()
    return weights.reindex(values.index)


def weighted_return(returns: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    available_weights = weights.where(returns.notna())
    numerator = (returns * available_weights).sum(axis=1, min_count=1)
    denominator = available_weights.sum(axis=1, min_count=1)
    return numerator / denominator


def build_targets(
    returns: pd.DataFrame, tvl: pd.DataFrame, categories: dict[str, str], common_protocols: list[str]
) -> pd.DataFrame:
    output = pd.DataFrame(index=returns.index)
    active = returns.notna().sum(axis=1)
    output["equal_weight_available"] = returns.mean(axis=1).where(active >= MIN_ACTIVE_PROTOCOLS)
    lagged_tvl = tvl.shift(1).where(returns.notna())
    output["lagged_tvl_weighted"] = weighted_return(returns, lagged_tvl).where(active >= MIN_ACTIVE_PROTOCOLS)
    capped = pd.DataFrame(index=lagged_tvl.index, columns=lagged_tvl.columns, dtype=float)
    for date in lagged_tvl.index:
        capped.loc[date] = cap_weights(lagged_tvl.loc[date]).to_numpy()
    output["capped_lagged_tvl_weighted"] = weighted_return(returns, capped).where(
        active >= MIN_ACTIVE_PROTOCOLS
    )

    category_weights = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    category_series = pd.Series(categories)
    for date in returns.index:
        available = returns.loc[date].dropna().index
        date_categories = category_series.reindex(available).dropna()
        number_categories = date_categories.nunique()
        if number_categories:
            counts = date_categories.value_counts()
            category_weights.loc[date, date_categories.index] = [
                1.0 / number_categories / counts[date_categories[protocol]] for protocol in date_categories.index
            ]
    output["category_balanced"] = weighted_return(returns, category_weights).where(
        active >= MIN_ACTIVE_PROTOCOLS
    )
    common_returns = returns[common_protocols]
    output["retrospective_common_history"] = common_returns.mean(axis=1).where(
        common_returns.notna().sum(axis=1) == len(common_protocols)
    )
    return output


def residualize_on_eth(
    returns: pd.DataFrame, eth_return: pd.Series, window: int = 180, min_obs: int = 90
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    residuals = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    betas = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    rows: list[dict] = []
    factor = eth_return.reindex(returns.index)
    lag_factor = factor.shift(1)
    for protocol in returns:
        series = returns[protocol]
        lag_series = series.shift(1)
        covariance = lag_series.rolling(window, min_periods=min_obs).cov(lag_factor)
        variance = lag_factor.rolling(window, min_periods=min_obs).var()
        beta = covariance / variance.replace(0.0, np.nan)
        alpha = lag_series.rolling(window, min_periods=min_obs).mean() - beta * lag_factor.rolling(
            window, min_periods=min_obs
        ).mean()
        residual = series - alpha - beta * factor
        betas[protocol] = beta
        residuals[protocol] = residual
        common = pd.concat([series, factor, residual], axis=1).dropna()
        rows.append(
            {
                "protocol_slug": protocol,
                "observations": len(common),
                "raw_eth_correlation": float(common.iloc[:, 0].corr(common.iloc[:, 1])),
                "median_past_only_eth_beta": float(beta.median()),
                "residual_to_raw_variance_ratio": float(common.iloc[:, 2].var() / common.iloc[:, 0].var()),
            }
        )
    return residuals, betas, pd.DataFrame(rows)


def fit_scaler(training: pd.DataFrame, columns: list[str]) -> dict[str, tuple[float, float]]:
    parameters = {}
    for column in columns:
        low = float(training[column].quantile(0.05))
        high = float(training[column].quantile(0.95))
        parameters[column] = (low, high if high > low else low + 1.0)
    return parameters


def apply_scaler(
    frame: pd.DataFrame, parameters: dict[str, tuple[float, float]], clip: bool = True
) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column, (low, high) in parameters.items():
        values = frame[column].clip(low, high) if clip else frame[column]
        output[column] = (values - low) / (high - low)
    return output


def score_models(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str], selected: tuple[str, ...] | None = None
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict]:
    scaler = fit_scaler(train, features + ["tvl_momentum"])
    train_scaled = apply_scaler(train, scaler)
    test_scaled = apply_scaler(test, scaler)
    weights = pd.Series(HEURISTIC_WEIGHTS)
    train_scores: dict[str, pd.Series] = {
        "original_heuristic": train_scaled[features].mul(weights).sum(axis=1),
        "equal_component_weight": train_scaled[features].mean(axis=1),
        "stablecoin_only": train_scaled["stablecoin_contraction"],
        "tvl_momentum": train_scaled["tvl_momentum"],
    }
    test_scores: dict[str, pd.Series] = {
        "original_heuristic": test_scaled[features].mul(weights).sum(axis=1),
        "equal_component_weight": test_scaled[features].mean(axis=1),
        "stablecoin_only": test_scaled["stablecoin_contraction"],
        "tvl_momentum": test_scaled["tvl_momentum"],
    }
    logistic: dict[str, dict] = {}
    for name, subset in [("unconstrained_logistic", tuple(features))]:
        intercept, coefficients, means, stds = fit_logistic(train[list(subset)], train["label"].astype(int))
        train_scores[name] = logistic_predict(train[list(subset)], intercept, coefficients, means, stds)
        test_scores[name] = logistic_predict(test[list(subset)], intercept, coefficients, means, stds)
        logistic[name] = {
            "intercept": intercept,
            "coefficients": coefficients.to_dict(),
            "means": means.to_dict(),
            "stds": stds.to_dict(),
            "features": list(subset),
        }
    if selected:
        intercept, coefficients, means, stds = fit_logistic(train[list(selected)], train["label"].astype(int))
        train_scores["parsimonious_selected"] = logistic_predict(
            train[list(selected)], intercept, coefficients, means, stds
        )
        test_scores["parsimonious_selected"] = logistic_predict(
            test[list(selected)], intercept, coefficients, means, stds
        )
        logistic["parsimonious_selected"] = {
            "intercept": intercept,
            "coefficients": coefficients.to_dict(),
            "means": means.to_dict(),
            "stds": stds.to_dict(),
            "features": list(selected),
        }
    return train_scores, test_scores, {"scaler": scaler, "logistic": logistic}


def inner_select_features(training: pd.DataFrame) -> tuple[tuple[str, ...], pd.DataFrame]:
    candidates = [(component,) for component in COMPONENTS]
    candidates += [
        ("stablecoin_contraction", component)
        for component in COMPONENTS
        if component != "stablecoin_contraction"
    ]
    candidates += [tuple(COMPONENTS)]
    start = max(240, int(len(training) * 0.50))
    boundaries = np.linspace(start, len(training), 4, dtype=int)
    rows: list[dict] = []
    for candidate in candidates:
        fold_values = []
        for fold in range(3):
            test = training.iloc[boundaries[fold] : boundaries[fold + 1]].copy()
            if test.empty:
                continue
            cutoff = test["date"].min() - pd.Timedelta(days=FORECAST_HORIZON_DAYS)
            train = training[training["date"] < cutoff].copy()
            if len(train) < 150 or train["label"].nunique() < 2 or test["label"].nunique() < 2:
                continue
            intercept, coefficients, means, stds = fit_logistic(train[list(candidate)], train["label"].astype(int))
            scores = logistic_predict(test[list(candidate)], intercept, coefficients, means, stds)
            fold_values.append(average_precision(test["label"].astype(int), scores))
        rows.append(
            {
                "features": "+".join(candidate),
                "feature_count": len(candidate),
                "mean_inner_average_precision": float(np.mean(fold_values)) if fold_values else np.nan,
                "folds": len(fold_values),
            }
        )
    results = pd.DataFrame(rows).dropna(subset=["mean_inner_average_precision"])
    best_value = float(results["mean_inner_average_precision"].max())
    eligible = results[results["mean_inner_average_precision"] >= best_value - 0.005]
    selected_row = eligible.sort_values(
        ["feature_count", "mean_inner_average_precision"], ascending=[True, False]
    ).iloc[0]
    selected = tuple(str(selected_row["features"]).split("+"))
    results["selected"] = results["features"].eq(selected_row["features"])
    return selected, results.sort_values("mean_inner_average_precision", ascending=False)


def block_bootstrap(y_true: pd.Series, scores: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 901)
    y = y_true.reset_index(drop=True)
    s = scores.reset_index(drop=True)
    n = len(y)
    block = min(BLOCK_BOOTSTRAP_LENGTH, n)
    values: list[float] = []
    for _ in range(BLOCK_BOOTSTRAP_REPLICATIONS):
        indices: list[int] = []
        while len(indices) < n:
            start = int(rng.integers(0, n - block + 1))
            indices.extend(range(start, start + block))
        sample = indices[:n]
        values.append(average_precision(y.iloc[sample], s.iloc[sample]))
    return pd.DataFrame(
        [
            {
                "metric": "average_precision",
                "point_estimate": average_precision(y, s),
                "ci_lower_95": float(np.quantile(values, 0.025)),
                "ci_upper_95": float(np.quantile(values, 0.975)),
                "replications": BLOCK_BOOTSTRAP_REPLICATIONS,
                "block_length_days": block,
            }
        ]
    )


def component_diagnostics(
    train: pd.DataFrame, test: pd.DataFrame, scaler: dict[str, tuple[float, float]]
) -> dict[str, pd.DataFrame]:
    correlation_rows = []
    vif_rows = []
    association_rows = []
    distribution_rows = []
    decile_rows = []
    for sample_name, sample in [("training", train), ("holdout", test)]:
        matrix = sample[COMPONENTS].corr()
        for left in COMPONENTS:
            for right in COMPONENTS:
                correlation_rows.append(
                    {"sample": sample_name, "component_1": left, "component_2": right, "correlation": matrix.loc[left, right]}
                )
        inverse = np.linalg.pinv(matrix.to_numpy(dtype=float))
        for index, component in enumerate(COMPONENTS):
            vif_rows.append({"sample": sample_name, "component": component, "vif": float(inverse[index, index])})
            association = sample[component].corr(sample["label"])
            association_rows.append(
                {
                    "sample": sample_name,
                    "component": component,
                    "component_event_correlation": association,
                    "average_precision": average_precision(sample["label"].astype(int), sample[component]),
                    "inverse_average_precision": average_precision(
                        sample["label"].astype(int), -sample[component]
                    ),
                }
            )
            for event, group in sample.groupby("label"):
                distribution_rows.append(
                    {
                        "sample": sample_name,
                        "component": component,
                        "event": int(event),
                        "observations": len(group),
                        "mean": float(group[component].mean()),
                        "median": float(group[component].median()),
                        "p25": float(group[component].quantile(0.25)),
                        "p75": float(group[component].quantile(0.75)),
                    }
                )
            ranked = sample[["date", "label", component]].copy()
            ranked["decile"] = pd.qcut(ranked[component].rank(method="first"), 10, labels=range(1, 11)).astype(int)
            grouped = ranked.groupby("decile", as_index=False).agg(
                observations=("label", "size"), event_rate=("label", "mean"), mean_component=(component, "mean")
            )
            grouped["sample"] = sample_name
            grouped["component"] = component
            decile_rows.extend(grouped.to_dict("records"))

    saturation_rows = []
    for component, (low, high) in scaler.items():
        if component not in COMPONENTS:
            continue
        for sample_name, sample in [("training", train), ("holdout", test)]:
            saturation_rows.append(
                {
                    "sample": sample_name,
                    "component": component,
                    "at_or_below_training_p05": float((sample[component] <= low).mean()),
                    "at_or_above_training_p95": float((sample[component] >= high).mean()),
                    "training_p05": low,
                    "training_p95": high,
                }
            )
    return {
        "component_correlations": pd.DataFrame(correlation_rows),
        "component_vif": pd.DataFrame(vif_rows),
        "component_associations": pd.DataFrame(association_rows),
        "component_distributions": pd.DataFrame(distribution_rows),
        "component_deciles": pd.DataFrame(decile_rows),
        "scaling_saturation": pd.DataFrame(saturation_rows),
    }


def ablation_tests(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    scaler = fit_scaler(train, COMPONENTS)
    train_scaled = apply_scaler(train, scaler)
    test_scaled = apply_scaler(test, scaler)
    rows: list[dict] = []
    specifications: list[tuple[str, list[str], str]] = []
    for component in COMPONENTS:
        specifications.append((f"single_{component}", [component], "single factor"))
    for component in COMPONENTS:
        if component != "stablecoin_contraction":
            specifications.append(
                (f"stablecoin_plus_{component}", ["stablecoin_contraction", component], "stablecoin plus one")
            )
    for omitted in COMPONENTS:
        specifications.append((f"heuristic_without_{omitted}", [x for x in COMPONENTS if x != omitted], "leave one out"))

    for name, subset, family in specifications:
        if family == "leave one out":
            local_weights = pd.Series({key: HEURISTIC_WEIGHTS[key] for key in subset})
            local_weights /= local_weights.sum()
            train_score = train_scaled[subset].mul(local_weights).sum(axis=1)
            test_score = test_scaled[subset].mul(local_weights).sum(axis=1)
        else:
            train_score = train_scaled[subset].mean(axis=1)
            test_score = test_scaled[subset].mean(axis=1)
        rows.append(
            {
                "specification": name,
                "family": family,
                "features": "+".join(subset),
                "training_average_precision": average_precision(train["label"].astype(int), train_score),
                "holdout_average_precision": average_precision(test["label"].astype(int), test_score),
            }
        )
    stable_ap = next(row["holdout_average_precision"] for row in rows if row["specification"] == "single_stablecoin_contraction")
    for row in rows:
        row["incremental_ap_vs_stablecoin"] = row["holdout_average_precision"] - stable_ap
    return pd.DataFrame(rows)


def coefficient_stability(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample_name, sample in [("training", train), ("holdout_diagnostic", test)]:
        intercept, coefficients, _, _ = fit_logistic(sample[COMPONENTS], sample["label"].astype(int))
        rows.append({"sample": sample_name, "component": "intercept", "coefficient": intercept})
        rows.extend(
            {"sample": sample_name, "component": component, "coefficient": float(value)}
            for component, value in coefficients.items()
        )
    output = pd.DataFrame(rows)
    pivot = output[output["component"] != "intercept"].pivot(index="component", columns="sample", values="coefficient")
    pivot["sign_stable"] = np.sign(pivot["training"]) == np.sign(pivot["holdout_diagnostic"])
    return output.merge(pivot["sign_stable"], left_on="component", right_index=True, how="left")


def incremental_logistic_tests(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    specifications = [("stablecoin_only", ("stablecoin_contraction",))]
    specifications += [
        (f"stablecoin_plus_{component}", ("stablecoin_contraction", component))
        for component in COMPONENTS
        if component != "stablecoin_contraction"
    ]
    specifications.append(("all_components", tuple(COMPONENTS)))
    rows = []
    for name, subset in specifications:
        intercept, coefficients, means, stds = fit_logistic(
            train[list(subset)], train["label"].astype(int)
        )
        train_score = logistic_predict(train[list(subset)], intercept, coefficients, means, stds)
        test_score = logistic_predict(test[list(subset)], intercept, coefficients, means, stds)
        rows.append(
            {
                "specification": name,
                "features": "+".join(subset),
                "training_average_precision": average_precision(train["label"], train_score),
                "holdout_average_precision": average_precision(test["label"], test_score),
                "holdout_roc_auc": roc_auc(test["label"], test_score),
                "coefficients": json.dumps(coefficients.to_dict(), sort_keys=True),
            }
        )
    stable_ap = rows[0]["holdout_average_precision"]
    for row in rows:
        row["incremental_ap_vs_stablecoin_logistic"] = row["holdout_average_precision"] - stable_ap
    return pd.DataFrame(rows)


def walk_forward(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    initial = max(500, int(len(frame) * 0.50))
    boundaries = np.linspace(initial, len(frame), 5, dtype=int)
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    selection_rows: list[pd.DataFrame] = []
    for fold in range(4):
        test = frame.iloc[boundaries[fold] : boundaries[fold + 1]].copy()
        cutoff = test["date"].min() - pd.Timedelta(days=FORECAST_HORIZON_DAYS)
        train = frame[frame["date"] < cutoff].copy()
        selected, selection = inner_select_features(train)
        selection["outer_fold"] = fold + 1
        selection_rows.append(selection)
        train_scores, test_scores, _ = score_models(train, test, COMPONENTS, selected)
        for model, test_score in test_scores.items():
            threshold = choose_threshold(train["label"].astype(int), train_scores[model])
            row = metric_row(test["label"].astype(int), test_score, threshold)
            metric_rows.append(
                {
                    "fold": fold + 1,
                    "model": model,
                    "selected_features": "+".join(selected) if model == "parsimonious_selected" else "",
                    "train_start": train["date"].min(),
                    "train_end": train["date"].max(),
                    "test_start": test["date"].min(),
                    "test_end": test["date"].max(),
                    "purge_days": FORECAST_HORIZON_DAYS,
                    "preprocessing_fit_end": train["date"].max(),
                    **row,
                }
            )
            prediction_rows.extend(
                {
                    "date": date,
                    "fold": fold + 1,
                    "model": model,
                    "label": int(label),
                    "score": float(score),
                }
                for date, label, score in zip(test["date"], test["label"], test_score)
            )
    predictions = pd.DataFrame(prediction_rows)
    aggregate_rows = []
    for model, sample in predictions.groupby("model"):
        aggregate_rows.append(
            {
                "fold": "aggregate_oof",
                "model": model,
                "observations": len(sample),
                "events": int(sample["label"].sum()),
                "prevalence": float(sample["label"].mean()),
                "average_precision": average_precision(sample["label"], sample["score"]),
                "roc_auc": roc_auc(sample["label"], sample["score"]),
                "fold_ap_mean": float(
                    pd.DataFrame(metric_rows).query("model == @model")["average_precision"].mean()
                ),
                "fold_ap_std": float(
                    pd.DataFrame(metric_rows).query("model == @model")["average_precision"].std(ddof=1)
                ),
            }
        )
    metrics = pd.concat([pd.DataFrame(metric_rows), pd.DataFrame(aggregate_rows)], ignore_index=True, sort=False)
    return metrics, predictions, pd.concat(selection_rows, ignore_index=True)


def rolling_performance(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ["original_heuristic", "stablecoin_only", "parsimonious_selected"]:
        sample = predictions[predictions["model"] == model].sort_values("date").reset_index(drop=True)
        for end in range(180, len(sample) + 1, 30):
            window = sample.iloc[end - 180 : end]
            rows.append(
                {
                    "model": model,
                    "window_start": window["date"].min(),
                    "window_end": window["date"].max(),
                    "observations": len(window),
                    "events": int(window["label"].sum()),
                    "prevalence": float(window["label"].mean()),
                    "average_precision": average_precision(window["label"], window["score"]),
                    "roc_auc": roc_auc(window["label"], window["score"]),
                    "rank_correlation": float(
                        window["score"].rank(method="average").corr(
                            window["label"].rank(method="average")
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def target_sensitivity(
    features: pd.DataFrame, targets: pd.DataFrame, holdout_start: pd.Timestamp
) -> pd.DataFrame:
    rows = []
    for target in TARGET_NAMES:
        label = stress_label(forward_sum(targets[target], 7))
        frame = features.copy()
        frame["label"] = label.reindex(frame.index)
        frame = frame.dropna(subset=COMPONENTS + ["tvl_momentum", "label"]).reset_index().rename(columns={"index": "date"})
        train = frame[frame["date"] < holdout_start - pd.Timedelta(days=7)].copy()
        test = frame[frame["date"] >= holdout_start].copy()
        selected, _ = inner_select_features(train)
        train_scores, test_scores, _ = score_models(train, test, COMPONENTS, selected)
        for model in ["original_heuristic", "stablecoin_only", "unconstrained_logistic", "parsimonious_selected"]:
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "selected_features": "+".join(selected) if model == "parsimonious_selected" else "",
                    "training_observations": len(train),
                    "holdout_observations": len(test),
                    "holdout_events": int(test["label"].sum()),
                    "holdout_prevalence": float(test["label"].mean()),
                    "average_precision": average_precision(test["label"].astype(int), test_scores[model]),
                    "roc_auc": roc_auc(test["label"].astype(int), test_scores[model]),
                }
            )
    return pd.DataFrame(rows)


def lag_horizon_tests(
    features: pd.DataFrame, target_return: pd.Series, holdout_start: pd.Timestamp
) -> pd.DataFrame:
    rows = []
    for horizon, lag in itertools.product([3, 7, 14], [0, 1, 3, 7]):
        label = stress_label(forward_sum(target_return, horizon))
        lagged = features[COMPONENTS + ["tvl_momentum"]].shift(lag).copy()
        lagged["label"] = label
        frame = lagged.dropna().reset_index().rename(columns={"index": "date"})
        train = frame[frame["date"] < holdout_start - pd.Timedelta(days=horizon)].copy()
        test = frame[frame["date"] >= holdout_start].copy()
        scaler = fit_scaler(train, COMPONENTS)
        train_scaled = apply_scaler(train, scaler)
        test_scaled = apply_scaler(test, scaler)
        weights = pd.Series(HEURISTIC_WEIGHTS)
        model_scores = {
            "original_heuristic": test_scaled[COMPONENTS].mul(weights).sum(axis=1),
            **{component: test_scaled[component] for component in COMPONENTS},
        }
        for model, score in model_scores.items():
            rows.append(
                {
                    "horizon_days": horizon,
                    "predictor_lag_days": lag,
                    "model": model,
                    "train_end": train["date"].max(),
                    "holdout_start": test["date"].min(),
                    "holdout_observations": len(test),
                    "holdout_events": int(test["label"].sum()),
                    "prevalence": float(test["label"].mean()),
                    "average_precision": average_precision(test["label"].astype(int), score),
                    "roc_auc": roc_auc(test["label"].astype(int), score),
                }
            )
    return pd.DataFrame(rows)


def episode_analysis(
    full_frame: pd.DataFrame,
    test: pd.DataFrame,
    returns: pd.DataFrame,
    target_return: pd.Series,
    score: pd.Series,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = test[["date", "label"]].copy()
    panel["score"] = score.to_numpy()
    panel["alert"] = (panel["score"] >= threshold).astype(int)
    starts = (panel["label"] == 1) & (panel["label"].shift(fill_value=0) == 0)
    panel["episode"] = starts.cumsum()
    episodes = []
    for episode, group in panel[panel["label"] == 1].groupby("episode"):
        onset = pd.Timestamp(group["date"].min())
        end = pd.Timestamp(group["date"].max())
        pre = panel[(panel["date"] >= onset - pd.Timedelta(days=7)) & (panel["date"] < onset)]
        during = group[group["date"] >= onset]
        pre_alert_dates = pre.loc[pre["alert"] == 1, "date"]
        lead = int((onset - pd.Timestamp(pre_alert_dates.max())).days) if len(pre_alert_dates) else np.nan
        driver_window = returns.loc[(returns.index > onset) & (returns.index <= onset + pd.Timedelta(days=7))]
        drivers = driver_window.sum(min_count=1).sort_values().head(3)
        forward_return = float(forward_sum(target_return, 7).get(onset, np.nan))
        episodes.append(
            {
                "episode": int(episode),
                "onset": onset,
                "end": end,
                "duration_days": len(group),
                "forward_7d_return_at_onset": forward_return,
                "severity_loss": 1.0 - math.exp(forward_return) if np.isfinite(forward_return) else np.nan,
                "pre_onset_alert": int(len(pre_alert_dates) > 0),
                "warning_lead_days": lead,
                "alert_on_or_after_onset": int(during["alert"].max()),
                "after_onset_only": int(len(pre_alert_dates) == 0 and during["alert"].max() == 1),
                "most_negative_protocols": "; ".join(f"{key}:{value:.3f}" for key, value in drivers.items()),
            }
        )
    episode_table = pd.DataFrame(episodes)

    panel["false_alert"] = ((panel["alert"] == 1) & (panel["label"] == 0)).astype(int)
    panel["false_group"] = ((panel["false_alert"] == 1) & (panel["false_alert"].shift(fill_value=0) == 0)).cumsum()
    false_rows = []
    for group_id, group in panel[panel["false_alert"] == 1].groupby("false_group"):
        false_rows.append(
            {
                "false_alert_episode": int(group_id),
                "start": group["date"].min(),
                "end": group["date"].max(),
                "duration_days": len(group),
                "maximum_score": float(group["score"].max()),
            }
        )

    study_rows = []
    onsets = episode_table["onset"].tolist()
    indexed = full_frame.set_index("date")
    for offset in range(-14, 15):
        observations = []
        for onset in onsets:
            date = pd.Timestamp(onset) + pd.Timedelta(days=offset)
            if date in indexed.index:
                row = indexed.loc[date]
                observations.append(row)
        if observations:
            sample = pd.DataFrame(observations)
            study_rows.append(
                {
                    "offset_days": offset,
                    "episodes_available": len(sample),
                    "mean_basket_daily_return": float(sample["equal_weight_available"].mean()),
                    **{f"mean_{component}": float(sample[component].mean()) for component in COMPONENTS},
                }
            )
    return episode_table, pd.DataFrame(false_rows), pd.DataFrame(study_rows)


def latest_edges(returns: pd.DataFrame, window: int, threshold: float) -> dict[str, list[tuple[str, float]]]:
    sample = returns.tail(window)
    correlation = sample.corr(min_periods=max(20, math.ceil(window * 0.75)))
    edge_map = {protocol: [] for protocol in returns.columns}
    for i, source in enumerate(returns.columns):
        for target in returns.columns[i + 1 :]:
            value = correlation.loc[source, target]
            if pd.notna(value) and value >= threshold:
                edge_map[source].append((target, float(value)))
                edge_map[target].append((source, float(value)))
    return edge_map


def run_scenario(
    returns: pd.DataFrame,
    current_tvl: pd.Series,
    multiplier: float,
    rounds: int,
    threshold: float,
    tail_cutoff: float,
    loss_cap: float,
    network_window: int,
    paths: int,
    seed: int,
    weighting: str = "current_tvl",
) -> dict[str, float | int | str]:
    protocols = list(returns.columns)
    if weighting == "equal":
        weights = pd.Series(1.0 / len(protocols), index=protocols)
    else:
        weights = current_tvl.reindex(protocols).fillna(0.0)
        weights /= weights.sum()
    seven_day_losses = (1.0 - np.exp(returns.rolling(7, min_periods=7).sum())).clip(0.0, loss_cap)
    tails = {}
    for protocol in protocols:
        positive = seven_day_losses[protocol].dropna()
        positive = positive[positive > 0]
        cutoff = positive.quantile(tail_cutoff)
        tails[protocol] = positive[positive >= cutoff].to_numpy()
    edges = latest_edges(returns, network_window, threshold)
    rng = np.random.default_rng(seed)
    losses = []
    affected = []
    for _ in range(paths):
        initial = str(rng.choice(protocols, p=weights.to_numpy()))
        shocks = {protocol: 0.0 for protocol in protocols}
        shocks[initial] = min(float(rng.choice(tails[initial])), loss_cap)
        visited = {initial}
        frontier = [initial]
        for _ in range(rounds):
            incoming: dict[str, list[float]] = {}
            for source in frontier:
                for target, correlation in edges[source]:
                    if target in visited:
                        continue
                    coefficient = multiplier * max(correlation, 0.0) * float(rng.uniform(0.75, 1.25))
                    propagated = min(shocks[source] * coefficient, loss_cap)
                    if propagated > 0:
                        incoming.setdefault(target, []).append(propagated)
            frontier = []
            for target, values in incoming.items():
                shocks[target] = min(1.0 - float(np.prod([1.0 - value for value in values])), loss_cap)
                visited.add(target)
                frontier.append(target)
        losses.append(float(sum(weights[p] * shocks[p] for p in protocols)))
        affected.append(sum(value > 0 for value in shocks.values()))
    values = pd.Series(losses)
    return {
        "propagation_multiplier": multiplier,
        "rounds": rounds,
        "edge_threshold": threshold,
        "initial_tail_cutoff": tail_cutoff,
        "loss_cap": loss_cap,
        "network_window_days": network_window,
        "weighting": weighting,
        "paths": paths,
        "mean_drawdown": float(values.mean()),
        "p95_drawdown": float(values.quantile(0.95)),
        "p99_drawdown": float(values.quantile(0.99)),
        "probability_over_10pct": float((values > 0.10).mean()),
        "mean_affected_protocols": float(np.mean(affected)),
    }


def scenario_grid(returns: pd.DataFrame, current_tvl: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        {
            "scenario": "baseline",
            **run_scenario(returns, current_tvl, 0.50, 2, 0.45, 0.85, 0.50, 60, 5000, RANDOM_SEED),
        }
    ]
    grid = itertools.product([0.30, 0.70], [1, 3], [0.30, 0.60], [0.80, 0.90], [0.30, 0.50], [30, 90])
    for scenario_id, values in enumerate(grid, start=1):
        multiplier, rounds, threshold, tail, cap, window = values
        rows.append(
            {
                "scenario": f"grid_{scenario_id:02d}",
                **run_scenario(
                    returns,
                    current_tvl,
                    multiplier,
                    rounds,
                    threshold,
                    tail,
                    cap,
                    window,
                    3000,
                    RANDOM_SEED + scenario_id,
                ),
            }
        )
    rows.append(
        {
            "scenario": "baseline_equal_weights",
            **run_scenario(
                returns, current_tvl, 0.50, 2, 0.45, 0.85, 0.50, 60, 5000, RANDOM_SEED + 500, "equal"
            ),
        }
    )
    scenarios = pd.DataFrame(rows)
    historical_losses = 1.0 - np.exp(returns.mean(axis=1).rolling(7, min_periods=7).sum())
    tail_cutoff = historical_losses.quantile(0.95)
    protocol_losses = 1.0 - np.exp(returns.rolling(7, min_periods=7).sum())
    tail_affected = (protocol_losses.loc[historical_losses >= tail_cutoff] > 0.01).sum(axis=1)
    comparison = pd.DataFrame(
        [
            {"quantity": "historical_7d_loss_p95", "value": float(historical_losses.quantile(0.95))},
            {"quantity": "historical_7d_loss_p99", "value": float(historical_losses.quantile(0.99))},
            {"quantity": "historical_7d_loss_max", "value": float(historical_losses.max())},
            {
                "quantity": "historical_tail_median_protocols_losing_over_1pct",
                "value": float(tail_affected.median()),
            },
            {"quantity": "scenario_grid_p95_min", "value": float(scenarios.query("scenario.str.startswith('grid')", engine="python")["p95_drawdown"].min())},
            {"quantity": "scenario_grid_p95_max", "value": float(scenarios.query("scenario.str.startswith('grid')", engine="python")["p95_drawdown"].max())},
            {"quantity": "scenario_grid_p99_min", "value": float(scenarios.query("scenario.str.startswith('grid')", engine="python")["p99_drawdown"].min())},
            {"quantity": "scenario_grid_p99_max", "value": float(scenarios.query("scenario.str.startswith('grid')", engine="python")["p99_drawdown"].max())},
        ]
    )
    return scenarios, comparison


def main() -> None:
    ensure_directories()
    long = pd.read_csv(PROCESSED_DIR / "protocol_returns.csv", parse_dates=["date"])
    system = pd.read_csv(PROCESSED_DIR / "system_daily.csv", parse_dates=["date"])
    eth = pd.read_csv(PROCESSED_DIR / "eth_market_control.csv", parse_dates=["date"]).set_index("date")
    returns = long.pivot(index="date", columns="protocol_slug", values="log_return").sort_index().clip(-0.35, 0.35)
    tvl = long.pivot(index="date", columns="protocol_slug", values="tvl_usd").sort_index()
    categories = long.drop_duplicates("protocol_slug").set_index("protocol_slug")["category"].to_dict()
    system_start = pd.Timestamp(system["date"].min())
    common_protocols = []
    for protocol in returns:
        sample = returns.loc[returns.index >= system_start, protocol]
        if returns[protocol].first_valid_index() <= system_start and sample.notna().mean() >= 0.95:
            common_protocols.append(protocol)

    targets = build_targets(returns, tvl, categories, common_protocols)
    network = build_network(returns, ROLLING_WINDOW_DAYS, EDGE_CORRELATION_THRESHOLD)
    residuals, betas, factor_protocol = residualize_on_eth(returns, eth["eth_log_return"])
    residual_network = build_network(residuals, ROLLING_WINDOW_DAYS, EDGE_CORRELATION_THRESHOLD).add_prefix("residual_")

    stable = system.set_index("date")[["stablecoin_supply_usd", "stablecoin_supply_log_flow"]]
    features = network.join(residual_network, how="outer").join(stable, how="left")
    features = features.join(targets, how="left")
    features["stablecoin_contraction"] = -features["stablecoin_supply_log_flow"].rolling(7, min_periods=7).sum()
    features["volatility"] = features["equal_weight_available"].rolling(30, min_periods=30).std()
    features["breadth"] = (returns < 0).sum(axis=1) / returns.notna().sum(axis=1)
    features["tvl_momentum"] = -features["equal_weight_available"].rolling(7, min_periods=7).sum()
    features["residual_volatility"] = residuals.mean(axis=1).rolling(30, min_periods=30).std()
    features["residual_breadth"] = (residuals < 0).sum(axis=1) / residuals.notna().sum(axis=1)
    features["residual_equal_weight_return"] = residuals.mean(axis=1)
    features["eth_log_return"] = eth["eth_log_return"].reindex(features.index)
    features["forward_return"] = forward_sum(features["equal_weight_available"], 7)
    features["label"] = stress_label(features["forward_return"])

    valid = features.dropna(subset=COMPONENTS + ["tvl_momentum", "label"]).reset_index().rename(columns={"index": "date"})
    split = make_split(valid)
    train = valid.loc[split.train_mask].copy()
    test = valid.loc[split.test_mask].copy()
    selected, final_inner_selection = inner_select_features(train)
    train_scores, test_scores, fitted = score_models(train, test, COMPONENTS, selected)

    validation_rows = []
    validation_predictions = test[["date", "label", "forward_return"]].copy()
    for model, test_score in test_scores.items():
        threshold = choose_threshold(train["label"].astype(int), train_scores[model])
        validation_rows.append(
            {
                "model": model,
                "selected_features": "+".join(selected) if model == "parsimonious_selected" else "",
                "threshold_origin": "training F2 optimum with 30% alert-rate cap",
                **metric_row(test["label"].astype(int), test_score, threshold),
            }
        )
        validation_predictions[f"score_{model}"] = test_score.to_numpy()
        validation_predictions[f"alert_{model}"] = (test_score >= threshold).astype(int).to_numpy()
    validation = pd.DataFrame(validation_rows)

    diagnostics = component_diagnostics(train, test, fitted["scaler"])
    ablations = ablation_tests(train, test)
    coefficients = coefficient_stability(train, test)
    incremental_logistic = incremental_logistic_tests(train, test)
    walk_metrics, walk_predictions, walk_selection = walk_forward(valid)
    rolling = rolling_performance(walk_predictions)
    target_results = target_sensitivity(features, targets, split.holdout_start)
    lag_results = lag_horizon_tests(features, targets["equal_weight_available"], split.holdout_start)

    primary_threshold = float(validation.set_index("model").loc["original_heuristic", "threshold"])
    episode_table, false_alerts, event_study = episode_analysis(
        valid,
        test,
        returns,
        targets["equal_weight_available"],
        test_scores["original_heuristic"],
        primary_threshold,
    )
    bootstrap = block_bootstrap(test["label"].astype(int), test_scores["original_heuristic"])

    raw_scaler = fit_scaler(train, COMPONENTS)
    clipped = apply_scaler(test, raw_scaler, clip=True)[COMPONENTS].mul(pd.Series(HEURISTIC_WEIGHTS)).sum(axis=1)
    unclipped = apply_scaler(test, raw_scaler, clip=False)[COMPONENTS].mul(pd.Series(HEURISTIC_WEIGHTS)).sum(axis=1)
    scaling_comparison = pd.DataFrame(
        [
            {"specification": "training_p05_p95_clipped", "average_precision": average_precision(test["label"], clipped), "roc_auc": roc_auc(test["label"], clipped)},
            {"specification": "same_linear_scaling_without_clipping", "average_precision": average_precision(test["label"], unclipped), "roc_auc": roc_auc(test["label"], unclipped)},
        ]
    )

    factor_rows = []
    for sample_name, sample in [("training", train), ("holdout", test)]:
        factor_rows.append(
            {
                "sample": sample_name,
                "raw_mean_correlation": float(sample["correlation"].mean()),
                "residual_mean_correlation": float(sample["residual_correlation"].mean()),
                "raw_mean_density": float(sample["density"].mean()),
                "residual_mean_density": float(sample["residual_density"].mean()),
                "correlation_attenuation": float(sample["correlation"].mean() - sample["residual_correlation"].mean()),
                "density_attenuation": float(sample["density"].mean() - sample["residual_density"].mean()),
            }
        )
    residual_components = pd.DataFrame(index=features.index)
    residual_components["correlation"] = features["residual_correlation"]
    residual_components["density"] = features["residual_density"]
    residual_components["stablecoin_contraction"] = features["stablecoin_contraction"]
    residual_components["volatility"] = features["residual_volatility"]
    residual_components["breadth"] = features["residual_breadth"]
    residual_components["label"] = features["label"]
    residual_valid = residual_components.dropna()
    residual_train = residual_valid[residual_valid.index < split.holdout_start - pd.Timedelta(days=7)]
    residual_test = residual_valid[residual_valid.index >= split.holdout_start]
    residual_scaler = fit_scaler(residual_train, COMPONENTS)
    residual_score = apply_scaler(residual_test, residual_scaler)[COMPONENTS].mul(pd.Series(HEURISTIC_WEIGHTS)).sum(axis=1)
    factor_model_results = pd.DataFrame(
        [
            {"specification": "raw_components_raw_target", "average_precision": average_precision(test["label"], test_scores["original_heuristic"]), "roc_auc": roc_auc(test["label"], test_scores["original_heuristic"])},
            {"specification": "eth_residual_components_raw_target", "average_precision": average_precision(residual_test["label"], residual_score), "roc_auc": roc_auc(residual_test["label"], residual_score)},
        ]
    )
    residual_target_label = stress_label(forward_sum(features["residual_equal_weight_return"], 7))
    residual_target = residual_components.drop(columns="label").copy()
    residual_target["label"] = residual_target_label
    residual_target = residual_target.dropna()
    residual_target_train = residual_target[
        residual_target.index < split.holdout_start - pd.Timedelta(days=7)
    ]
    residual_target_test = residual_target[residual_target.index >= split.holdout_start]
    residual_target_scaler = fit_scaler(residual_target_train, COMPONENTS)
    residual_target_scaled_train = apply_scaler(residual_target_train, residual_target_scaler)
    residual_target_scaled_test = apply_scaler(residual_target_test, residual_target_scaler)
    residual_target_score = residual_target_scaled_test[COMPONENTS].mul(
        pd.Series(HEURISTIC_WEIGHTS)
    ).sum(axis=1)
    residual_stablecoin_score = residual_target_scaled_test["stablecoin_contraction"]
    factor_model_results = pd.concat(
        [
            factor_model_results,
            pd.DataFrame(
                [
                    {
                        "specification": "eth_residual_components_residual_target",
                        "average_precision": average_precision(
                            residual_target_test["label"], residual_target_score
                        ),
                        "roc_auc": roc_auc(residual_target_test["label"], residual_target_score),
                    },
                    {
                        "specification": "stablecoin_only_residual_target",
                        "average_precision": average_precision(
                            residual_target_test["label"], residual_stablecoin_score
                        ),
                        "roc_auc": roc_auc(
                            residual_target_test["label"], residual_stablecoin_score
                        ),
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    current_tvl = tvl.ffill().iloc[-1]
    scenarios, scenario_comparison = scenario_grid(returns.dropna(how="all"), current_tvl)

    target_daily = targets.copy()
    for target in TARGET_NAMES:
        target_daily[f"forward_7d_{target}"] = forward_sum(target_daily[target], 7)
        target_daily[f"label_7d_{target}"] = stress_label(target_daily[f"forward_7d_{target}"])
    target_daily.reset_index().to_csv(MODELS_DIR / "target_definitions_daily.csv", index=False)
    features.reset_index().to_csv(MODELS_DIR / "analysis_panel.csv", index=False)
    factor_protocol.to_csv(MODELS_DIR / "eth_factor_protocol_summary.csv", index=False)
    pd.DataFrame(factor_rows).to_csv(MODELS_DIR / "eth_factor_network_summary.csv", index=False)
    factor_model_results.to_csv(MODELS_DIR / "eth_factor_model_comparison.csv", index=False)
    validation.to_csv(MODELS_DIR / "validation_metrics.csv", index=False)
    validation_predictions.to_csv(MODELS_DIR / "validation_predictions.csv", index=False)
    final_inner_selection.to_csv(MODELS_DIR / "final_training_model_selection.csv", index=False)
    ablations.to_csv(MODELS_DIR / "component_ablation_incremental.csv", index=False)
    coefficients.to_csv(MODELS_DIR / "component_coefficient_stability.csv", index=False)
    incremental_logistic.to_csv(MODELS_DIR / "incremental_logistic_tests.csv", index=False)
    for name, output in diagnostics.items():
        output.to_csv(MODELS_DIR / f"{name}.csv", index=False)
    scaling_comparison.to_csv(MODELS_DIR / "scaling_method_comparison.csv", index=False)
    walk_metrics.to_csv(MODELS_DIR / "walk_forward_metrics.csv", index=False)
    walk_predictions.to_csv(MODELS_DIR / "walk_forward_predictions.csv", index=False)
    walk_selection.to_csv(MODELS_DIR / "walk_forward_model_selection.csv", index=False)
    rolling.to_csv(MODELS_DIR / "rolling_performance.csv", index=False)
    target_results.to_csv(MODELS_DIR / "target_definition_sensitivity.csv", index=False)
    lag_results.to_csv(MODELS_DIR / "lag_horizon_performance.csv", index=False)
    episode_table.to_csv(MODELS_DIR / "episode_details.csv", index=False)
    false_alerts.to_csv(MODELS_DIR / "false_alert_episodes.csv", index=False)
    event_study.to_csv(MODELS_DIR / "episode_event_study.csv", index=False)
    bootstrap.to_csv(MODELS_DIR / "bootstrap_confidence_intervals.csv", index=False)
    scenarios.to_csv(MODELS_DIR / "scenario_parameter_sensitivity.csv", index=False)
    scenario_comparison.to_csv(MODELS_DIR / "scenario_historical_comparison.csv", index=False)
    pd.DataFrame(
        [
            {"assumption": "initial protocol", "value": "sampled by current TVL weight", "status": "current exposure calibration"},
            {"assumption": "initial shock", "value": "protocol historical seven-day loss tail", "status": "empirical distribution"},
            {"assumption": "tail cutoff", "value": "0.80, 0.85, or 0.90", "status": "heuristic sensitivity"},
            {"assumption": "edge", "value": "current rolling TVL-return correlation above threshold", "status": "statistical association, not exposure"},
            {"assumption": "propagation multiplier", "value": "0.30, 0.50, or 0.70", "status": "heuristic sensitivity"},
            {"assumption": "random multiplier", "value": "Uniform(0.75, 1.25)", "status": "heuristic dispersion"},
            {"assumption": "rounds", "value": "1, 2, or 3", "status": "heuristic sensitivity"},
            {"assumption": "loss cap", "value": "0.30 or 0.50", "status": "heuristic sensitivity"},
            {"assumption": "network window", "value": "30, 60, or 90 days", "status": "heuristic sensitivity"},
            {"assumption": "simulation paths", "value": "3,000 grid; 5,000 baseline", "status": "numerical sampling choice"},
        ]
    ).to_csv(MODELS_DIR / "scenario_assumptions.csv", index=False)

    validation_index = validation.set_index("model")
    walk_aggregate = walk_metrics[walk_metrics["fold"].astype(str) == "aggregate_oof"].set_index("model")
    target_pivot = target_results.pivot(index="target", columns="model", values="average_precision")
    lag_stable = lag_results[lag_results["model"] == "stablecoin_contraction"]
    summary = {
        "research_title": "Aggregate On-Chain Liquidity Indicators and Future DeFi Stress",
        "sample_start": str(valid["date"].min().date()),
        "sample_end": str(valid["date"].max().date()),
        "train_end": str(split.train_end.date()),
        "holdout_start": str(split.holdout_start.date()),
        "holdout_end": str(test["date"].max().date()),
        "training_observations": int(len(train)),
        "holdout_observations": int(len(test)),
        "holdout_events": int(test["label"].sum()),
        "holdout_prevalence": float(test["label"].mean()),
        "original_holdout_average_precision": float(validation_index.loc["original_heuristic", "average_precision"]),
        "original_holdout_roc_auc": float(validation_index.loc["original_heuristic", "roc_auc"]),
        "stablecoin_holdout_average_precision": float(validation_index.loc["stablecoin_only", "average_precision"]),
        "stablecoin_holdout_roc_auc": float(validation_index.loc["stablecoin_only", "roc_auc"]),
        "full_logistic_holdout_average_precision": float(validation_index.loc["unconstrained_logistic", "average_precision"]),
        "parsimonious_features": list(selected),
        "parsimonious_holdout_average_precision": float(validation_index.loc["parsimonious_selected", "average_precision"]),
        "walk_forward_original_average_precision": float(walk_aggregate.loc["original_heuristic", "average_precision"]),
        "walk_forward_stablecoin_average_precision": float(walk_aggregate.loc["stablecoin_only", "average_precision"]),
        "walk_forward_parsimonious_average_precision": float(walk_aggregate.loc["parsimonious_selected", "average_precision"]),
        "correlation_density_training_correlation": float(train[["correlation", "density"]].corr().iloc[0, 1]),
        "correlation_density_holdout_correlation": float(test[["correlation", "density"]].corr().iloc[0, 1]),
        "raw_holdout_mean_correlation": float(test["correlation"].mean()),
        "residual_holdout_mean_correlation": float(test["residual_correlation"].mean()),
        "raw_holdout_mean_density": float(test["density"].mean()),
        "residual_holdout_mean_density": float(test["residual_density"].mean()),
        "episodes": int(len(episode_table)),
        "episodes_with_pre_onset_alert": int(episode_table["pre_onset_alert"].sum()),
        "episodes_after_onset_only": int(episode_table["after_onset_only"].sum()),
        "false_alert_episodes": int(len(false_alerts)),
        "median_warning_lead_days": float(episode_table["warning_lead_days"].median()),
        "scenario_p95_min": float(scenario_comparison.set_index("quantity").loc["scenario_grid_p95_min", "value"]),
        "scenario_p95_max": float(scenario_comparison.set_index("quantity").loc["scenario_grid_p95_max", "value"]),
        "scenario_p99_min": float(scenario_comparison.set_index("quantity").loc["scenario_grid_p99_min", "value"]),
        "scenario_p99_max": float(scenario_comparison.set_index("quantity").loc["scenario_grid_p99_max", "value"]),
        "historical_7d_loss_p99": float(scenario_comparison.set_index("quantity").loc["historical_7d_loss_p99", "value"]),
        "target_stablecoin_ap_min": float(target_pivot["stablecoin_only"].min()),
        "target_stablecoin_ap_max": float(target_pivot["stablecoin_only"].max()),
        "stablecoin_lag0_h7_ap": float(lag_stable.query("horizon_days == 7 and predictor_lag_days == 0")["average_precision"].iloc[0]),
        "stablecoin_lag7_h7_ap": float(lag_stable.query("horizon_days == 7 and predictor_lag_days == 7")["average_precision"].iloc[0]),
        "common_history_protocols": common_protocols,
        "manual_selected_universe": list(returns.columns),
        "fixed_return_bounds": [-0.35, 0.35],
        "component_weights": HEURISTIC_WEIGHTS,
        "display_bands_prespecified": [55, 70],
        "market_control_source": "DefiLlama Coins API, coingecko:ethereum",
        "market_control_retrieved": "2026-07-20",
    }
    (MODELS_DIR / "model_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (MODELS_DIR / "calibration_parameters.json").write_text(json.dumps(fitted, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"Deep analysis complete: holdout AP original={summary['original_holdout_average_precision']:.3f}, "
        f"stablecoin={summary['stablecoin_holdout_average_precision']:.3f}; "
        f"walk-forward AP stablecoin={summary['walk_forward_stablecoin_average_precision']:.3f}."
    )


if __name__ == "__main__":
    main()
