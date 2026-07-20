from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from config import FIGURES_DIR, MODELS_DIR, ensure_directories


COLORS = {
    "ink": "#1f2933",
    "muted": "#64717d",
    "grid": "#d9dee3",
    "blue": "#2457a6",
    "red": "#b54a4a",
    "green": "#277a68",
    "gold": "#b7791f",
    "purple": "#7256a3",
    "light": "#f3f5f7",
}


def rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def canvas(title: str, subtitle: str, width: int = 1500, height: int = 800) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 35), title, font=font(36, True), fill=rgb(COLORS["ink"]))
    draw.text((70, 82), subtitle, font=font(20), fill=rgb(COLORS["muted"]))
    return image, draw


def save(image: Image.Image, name: str) -> None:
    image.save(FIGURES_DIR / name, quality=96)


def draw_axes(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    y_ticks: list[float],
    y_format,
    y_min: float = 0.0,
    y_max: float = 1.0,
) -> None:
    left, top, right, bottom = box
    for value in y_ticks:
        y = bottom - int((value - y_min) / (y_max - y_min) * (bottom - top))
        draw.line((left, y, right, y), fill=rgb(COLORS["grid"]), width=1)
        draw.text((left - 58, y - 10), y_format(value), font=font(17), fill=rgb(COLORS["muted"]))
    draw.line((left, top, left, bottom), fill=rgb(COLORS["ink"]), width=2)
    draw.line((left, bottom, right, bottom), fill=rgb(COLORS["ink"]), width=2)


def figure_factor_adjustment(panel: pd.DataFrame) -> None:
    image, draw = canvas(
        "Most TVL-return co-movement disappears after ETH adjustment",
        "Sixty-day mean correlation and threshold density; ETH betas use only the preceding 180 days",
    )
    weekly = panel.set_index("date")[["correlation", "residual_correlation", "density", "residual_density"]].resample("14D").mean().dropna()
    boxes = [(105, 155, 720, 690), (840, 155, 1455, 690)]
    panels = [
        ("Mean pairwise correlation", "correlation", "residual_correlation"),
        ("Edge density", "density", "residual_density"),
    ]
    for box, (label, raw, adjusted) in zip(boxes, panels):
        left, top, right, bottom = box
        draw.text((left, 125), label, font=font(22, True), fill=rgb(COLORS["ink"]))
        draw_axes(draw, box, [0.0, 0.25, 0.50, 0.75, 1.0], lambda x: f"{x:.2f}")
        dates = weekly.index
        span = max((dates.max() - dates.min()).days, 1)
        for column, color in [(raw, COLORS["blue"]), (adjusted, COLORS["red"])]:
            values = weekly[column].clip(0.0, 1.0)
            points = [
                (
                    left + int((date - dates.min()).days / span * (right - left)),
                    bottom - int(value * (bottom - top)),
                )
                for date, value in values.items()
            ]
            draw.line(points, fill=rgb(color), width=4)
        for fraction in [0.0, 0.5, 1.0]:
            date = dates.min() + (dates.max() - dates.min()) * fraction
            x = left + int(fraction * (right - left))
            draw.text((x - 30, bottom + 18), date.strftime("%Y-%m"), font=font(16), fill=rgb(COLORS["muted"]))
    draw.line((565, 746, 610, 746), fill=rgb(COLORS["blue"]), width=5)
    draw.text((620, 735), "Raw USD TVL returns", font=font(18), fill=rgb(COLORS["ink"]))
    draw.line((830, 746, 875, 746), fill=rgb(COLORS["red"]), width=5)
    draw.text((885, 735), "ETH-adjusted residuals", font=font(18), fill=rgb(COLORS["ink"]))
    save(image, "figure_1_eth_factor_adjustment.png")


def heat_color(value: float) -> tuple[int, int, int]:
    value = float(np.clip(value, -1.0, 1.0))
    if value >= 0:
        start, end = np.array([245, 247, 249]), np.array(rgb(COLORS["blue"]))
    else:
        start, end = np.array([245, 247, 249]), np.array(rgb(COLORS["red"]))
    return tuple((start + abs(value) * (end - start)).astype(int))


def figure_component_correlation(correlations: pd.DataFrame) -> None:
    image, draw = canvas(
        "Correlation and density duplicate the same information",
        "Pearson correlations among raw predictor components in training and final holdout samples",
    )
    labels = ["Correlation", "Density", "Stablecoin", "Volatility", "Breadth"]
    components = ["correlation", "density", "stablecoin_contraction", "volatility", "breadth"]
    for panel_index, sample in enumerate(["training", "holdout"]):
        matrix = correlations[correlations["sample"] == sample].pivot(
            index="component_1", columns="component_2", values="correlation"
        ).reindex(index=components, columns=components)
        origin_x = 105 + panel_index * 735
        origin_y = 205
        cell = 92
        draw.text((origin_x, 135), sample.title(), font=font(25, True), fill=rgb(COLORS["ink"]))
        for i, label in enumerate(labels):
            draw.text((origin_x - 98, origin_y + i * cell + 34), label, font=font(15), fill=rgb(COLORS["ink"]))
            draw.text((origin_x + i * cell + 6, origin_y - 34), label[:7], font=font(14), fill=rgb(COLORS["ink"]))
        for row in range(5):
            for column in range(5):
                value = float(matrix.iloc[row, column])
                x0, y0 = origin_x + column * cell, origin_y + row * cell
                draw.rectangle((x0, y0, x0 + cell - 3, y0 + cell - 3), fill=heat_color(value))
                text_color = "#ffffff" if abs(value) > 0.60 else COLORS["ink"]
                draw.text((x0 + 20, y0 + 31), f"{value:.2f}", font=font(18, True), fill=rgb(text_color))
    save(image, "figure_2_component_correlation.png")


def figure_deciles(deciles: pd.DataFrame) -> None:
    image, draw = canvas(
        "Holdout event rates do not rise with the original risk direction",
        "Event prevalence within holdout deciles; decile 10 contains the highest component values",
    )
    box = (120, 155, 1430, 680)
    draw_axes(draw, box, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], lambda x: f"{x:.0%}", 0.0, 0.5)
    holdout = deciles[deciles["sample"] == "holdout"]
    series = [
        ("correlation", "Correlation", COLORS["blue"]),
        ("stablecoin_contraction", "Stablecoin contraction", COLORS["green"]),
        ("volatility", "Volatility", COLORS["red"]),
        ("breadth", "Breadth", COLORS["gold"]),
    ]
    left, top, right, bottom = box
    for component, label, color in series:
        sample = holdout[holdout["component"] == component].sort_values("decile")
        points = []
        for row in sample.itertuples():
            x = left + int((row.decile - 1) / 9 * (right - left))
            y = bottom - int(float(row.event_rate) / 0.5 * (bottom - top))
            points.append((x, y))
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=rgb(color))
        draw.line(points, fill=rgb(color), width=4)
    prevalence_y = bottom - int(0.20 / 0.5 * (bottom - top))
    draw.line((left, prevalence_y, right, prevalence_y), fill=rgb(COLORS["muted"]), width=2)
    for decile in range(1, 11):
        x = left + int((decile - 1) / 9 * (right - left))
        draw.text((x - 6, bottom + 20), str(decile), font=font(17), fill=rgb(COLORS["muted"]))
    draw.text((715, 720), "Component decile", font=font(19), fill=rgb(COLORS["muted"]))
    legend_x = 170
    for index, (_, label, color) in enumerate(series):
        x = legend_x + index * 285
        draw.line((x, 754, x + 35, 754), fill=rgb(color), width=5)
        draw.text((x + 45, 743), label, font=font(17), fill=rgb(COLORS["ink"]))
    save(image, "figure_3_component_deciles.png")


def figure_model_performance(validation: pd.DataFrame, walk: pd.DataFrame) -> None:
    image, draw = canvas(
        "Model rankings depend on sign flexibility and validation period",
        "Average precision in the final holdout and aggregated expanding-window out-of-fold predictions",
    )
    models = ["original_heuristic", "stablecoin_only", "unconstrained_logistic", "parsimonious_selected"]
    labels = ["Original\nheuristic", "Stablecoin\nonly", "Full\nlogistic", "Selected\nparsimonious"]
    holdout = validation.set_index("model")["average_precision"]
    oof = walk[walk["fold"].astype(str) == "aggregate_oof"].set_index("model")["average_precision"]
    box = (140, 155, 1415, 680)
    draw_axes(draw, box, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], lambda x: f"{x:.1f}", 0.0, 0.5)
    left, top, right, bottom = box
    group_width = (right - left) / len(models)
    for index, (model, label) in enumerate(zip(models, labels)):
        center = left + (index + 0.5) * group_width
        for offset, value, color in [(-40, holdout[model], COLORS["blue"]), (40, oof[model], COLORS["red"])]:
            x0 = int(center + offset - 33)
            x1 = int(center + offset + 33)
            y = bottom - int(value / 0.5 * (bottom - top))
            draw.rectangle((x0, y, x1, bottom), fill=rgb(color))
            draw.text((x0 + 5, y - 27), f"{value:.3f}", font=font(16, True), fill=rgb(COLORS["ink"]))
        line1, line2 = label.split("\n")
        draw.text((int(center - 48), bottom + 15), line1, font=font(17), fill=rgb(COLORS["ink"]))
        draw.text((int(center - 48), bottom + 37), line2, font=font(17), fill=rgb(COLORS["ink"]))
    draw.rectangle((535, 748, 555, 768), fill=rgb(COLORS["blue"]))
    draw.text((568, 745), "Final holdout", font=font(17), fill=rgb(COLORS["ink"]))
    draw.rectangle((760, 748, 780, 768), fill=rgb(COLORS["red"]))
    draw.text((793, 745), "Walk-forward OOF", font=font(17), fill=rgb(COLORS["ink"]))
    save(image, "figure_4_model_performance.png")


def figure_walk_forward(walk: pd.DataFrame) -> None:
    image, draw = canvas(
        "Predictive ranking changes sharply across chronological folds",
        "Fold-specific average precision; dotted grey series is fold event prevalence",
    )
    box = (135, 155, 1415, 680)
    draw_axes(draw, box, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], lambda x: f"{x:.1f}", 0.0, 0.5)
    folds = walk[pd.to_numeric(walk["fold"], errors="coerce").notna()].copy()
    folds["fold"] = folds["fold"].astype(int)
    series = [
        ("original_heuristic", "Original heuristic", COLORS["blue"]),
        ("stablecoin_only", "Stablecoin only", COLORS["green"]),
        ("parsimonious_selected", "Selected parsimonious", COLORS["red"]),
    ]
    left, top, right, bottom = box
    for model, label, color in series:
        sample = folds[folds["model"] == model].sort_values("fold")
        points = []
        for row in sample.itertuples():
            x = left + int((row.fold - 1) / 3 * (right - left))
            y = bottom - int(row.average_precision / 0.5 * (bottom - top))
            points.append((x, y))
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=rgb(color))
        draw.line(points, fill=rgb(color), width=5)
    prevalence = folds[folds["model"] == "original_heuristic"].sort_values("fold")
    prevalence_points = []
    for row in prevalence.itertuples():
        x = left + int((row.fold - 1) / 3 * (right - left))
        y = bottom - int(row.prevalence / 0.5 * (bottom - top))
        prevalence_points.append((x, y))
    draw.line(prevalence_points, fill=rgb(COLORS["muted"]), width=3)
    for fold in range(1, 5):
        x = left + int((fold - 1) / 3 * (right - left))
        draw.text((x - 30, bottom + 22), f"Fold {fold}", font=font(18), fill=rgb(COLORS["ink"]))
    for index, (_, label, color) in enumerate(series + [("prevalence", "Prevalence", COLORS["muted"])]):
        x = 220 + index * 300
        draw.line((x, 750, x + 38, 750), fill=rgb(color), width=5)
        draw.text((x + 48, 739), label, font=font(17), fill=rgb(COLORS["ink"]))
    save(image, "figure_5_walk_forward.png")


def figure_lag_timing(lag_results: pd.DataFrame) -> None:
    image, draw = canvas(
        "Stablecoin information weakens as the observation is moved back",
        "Seven-day outcome, final holdout; average precision for predictors observed at t, t-1, t-3, and t-7",
    )
    box = (135, 155, 1415, 680)
    draw_axes(draw, box, [0.1, 0.15, 0.2, 0.25, 0.3], lambda x: f"{x:.2f}", 0.1, 0.3)
    sample = lag_results[lag_results["horizon_days"] == 7]
    series = [
        ("original_heuristic", "Original heuristic", COLORS["blue"]),
        ("stablecoin_contraction", "Stablecoin contraction", COLORS["green"]),
        ("correlation", "Correlation (hypothesized sign)", COLORS["red"]),
    ]
    left, top, right, bottom = box
    y_min, y_max = 0.10, 0.30
    for model, label, color in series:
        values = sample[sample["model"] == model].sort_values("predictor_lag_days")
        points = []
        for index, row in enumerate(values.itertuples()):
            x = left + int(index / 3 * (right - left))
            y = bottom - int((row.average_precision - y_min) / (y_max - y_min) * (bottom - top))
            points.append((x, y))
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=rgb(color))
        draw.line(points, fill=rgb(color), width=5)
    prevalence = float(sample["prevalence"].iloc[0])
    y = bottom - int((prevalence - y_min) / (y_max - y_min) * (bottom - top))
    draw.line((left, y, right, y), fill=rgb(COLORS["muted"]), width=2)
    for index, lag in enumerate([0, 1, 3, 7]):
        x = left + int(index / 3 * (right - left))
        draw.text((x - 24, bottom + 22), f"t-{lag}" if lag else "t", font=font(18), fill=rgb(COLORS["ink"]))
    for index, (_, label, color) in enumerate(series):
        x = 210 + index * 390
        draw.line((x, 750, x + 38, 750), fill=rgb(color), width=5)
        draw.text((x + 48, 739), label, font=font(17), fill=rgb(COLORS["ink"]))
    save(image, "figure_6_lag_timing.png")


def figure_episode_timing(event_study: pd.DataFrame, panel: pd.DataFrame, holdout_start: pd.Timestamp) -> None:
    image, draw = canvas(
        "Correlation and volatility are weakest near labelled episode onset",
        "Mean predictor paths around holdout onsets, expressed in training-sample standard-deviation units",
    )
    box = (135, 155, 1415, 680)
    left, top, right, bottom = box
    y_min, y_max = -1.5, 1.5
    for value in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
        y = bottom - int((value - y_min) / (y_max - y_min) * (bottom - top))
        draw.line((left, y, right, y), fill=rgb(COLORS["grid"]), width=1)
        draw.text((left - 55, y - 10), f"{value:.1f}", font=font(16), fill=rgb(COLORS["muted"]))
    draw.line((left, top, left, bottom), fill=rgb(COLORS["ink"]), width=2)
    draw.line((left, bottom, right, bottom), fill=rgb(COLORS["ink"]), width=2)
    series = [
        ("correlation", "Correlation", COLORS["blue"]),
        ("stablecoin_contraction", "Stablecoin contraction", COLORS["green"]),
        ("volatility", "Volatility", COLORS["red"]),
        ("breadth", "Breadth", COLORS["gold"]),
    ]
    training = panel[panel["date"] < holdout_start]
    for component, label, color in series:
        mean = float(training[component].mean())
        std = float(training[component].std(ddof=0))
        values = (event_study[f"mean_{component}"] - mean) / std
        points = []
        for offset, value in zip(event_study["offset_days"], values):
            x = left + int((offset + 14) / 28 * (right - left))
            y = bottom - int((value - y_min) / (y_max - y_min) * (bottom - top))
            points.append((x, y))
        draw.line(points, fill=rgb(color), width=4)
    onset_x = left + int(14 / 28 * (right - left))
    draw.line((onset_x, top, onset_x, bottom), fill=rgb(COLORS["ink"]), width=3)
    draw.text((onset_x + 10, top + 8), "Labelled onset", font=font(17, True), fill=rgb(COLORS["ink"]))
    for offset in [-14, -7, 0, 7, 14]:
        x = left + int((offset + 14) / 28 * (right - left))
        draw.text((x - 15, bottom + 22), str(offset), font=font(17), fill=rgb(COLORS["muted"]))
    for index, (_, label, color) in enumerate(series):
        x = 205 + index * 290
        draw.line((x, 750, x + 38, 750), fill=rgb(color), width=5)
        draw.text((x + 48, 739), label, font=font(17), fill=rgb(COLORS["ink"]))
    save(image, "figure_7_episode_timing.png")


def figure_scenario_sensitivity(scenarios: pd.DataFrame, comparison: pd.DataFrame) -> None:
    image, draw = canvas(
        "Scenario tails are dominated by structural assumptions",
        "p95 and p99 conditional drawdowns across 64 propagation parameter combinations",
    )
    grid = scenarios[scenarios["scenario"].str.startswith("grid")].sort_values("p95_drawdown").reset_index(drop=True)
    box = (135, 155, 1415, 680)
    left, top, right, bottom = box
    y_max = 0.36
    draw_axes(draw, box, [0.0, 0.1, 0.2, 0.3], lambda x: f"{x:.0%}", 0.0, y_max)
    p95_points, p99_points = [], []
    for index, row in grid.iterrows():
        x = left + int(index / (len(grid) - 1) * (right - left))
        p95_points.append((x, bottom - int(row["p95_drawdown"] / y_max * (bottom - top))))
        p99_points.append((x, bottom - int(row["p99_drawdown"] / y_max * (bottom - top))))
    draw.line(p95_points, fill=rgb(COLORS["blue"]), width=4)
    draw.line(p99_points, fill=rgb(COLORS["red"]), width=4)
    lookup = comparison.set_index("quantity")["value"]
    historical = float(lookup["historical_7d_loss_p99"])
    y = bottom - int(historical / y_max * (bottom - top))
    draw.line((left, y, right, y), fill=rgb(COLORS["muted"]), width=3)
    draw.text((right - 240, y - 25), "Historical 7-day p99", font=font(16), fill=rgb(COLORS["muted"]))
    draw.text((650, bottom + 24), "Parameter combinations ordered by p95", font=font(18), fill=rgb(COLORS["muted"]))
    for index, (label, color) in enumerate([("Scenario p95", COLORS["blue"]), ("Scenario p99", COLORS["red"]), ("Historical p99", COLORS["muted"])]):
        x = 355 + index * 275
        draw.line((x, 750, x + 38, 750), fill=rgb(color), width=5)
        draw.text((x + 48, 739), label, font=font(17), fill=rgb(COLORS["ink"]))
    save(image, "figure_8_scenario_sensitivity.png")


def main() -> None:
    ensure_directories()
    panel = pd.read_csv(MODELS_DIR / "analysis_panel.csv", parse_dates=["date"])
    correlations = pd.read_csv(MODELS_DIR / "component_correlations.csv")
    deciles = pd.read_csv(MODELS_DIR / "component_deciles.csv")
    validation = pd.read_csv(MODELS_DIR / "validation_metrics.csv")
    walk = pd.read_csv(MODELS_DIR / "walk_forward_metrics.csv")
    lag_results = pd.read_csv(MODELS_DIR / "lag_horizon_performance.csv")
    event_study = pd.read_csv(MODELS_DIR / "episode_event_study.csv")
    scenarios = pd.read_csv(MODELS_DIR / "scenario_parameter_sensitivity.csv")
    comparison = pd.read_csv(MODELS_DIR / "scenario_historical_comparison.csv")

    figure_factor_adjustment(panel)
    figure_component_correlation(correlations)
    figure_deciles(deciles)
    figure_model_performance(validation, walk)
    figure_walk_forward(walk)
    figure_lag_timing(lag_results)
    holdout_start = pd.Timestamp(json.loads((MODELS_DIR / "model_summary.json").read_text())["holdout_start"])
    figure_episode_timing(event_study, panel, holdout_start)
    figure_scenario_sensitivity(scenarios, comparison)
    print("Created eight evidence-focused figures.")


if __name__ == "__main__":
    main()
