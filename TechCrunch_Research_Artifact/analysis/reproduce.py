"""Estimate OLS models and plot predictions from video-level measures.

Run from any directory: python analysis/reproduce.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


ROOT = Path(__file__).resolve().parents[1]
CONTROLS = [
    "Pocket_Hands_Ratio", "Hand Gesture Frequency", "Body_Openness",
    "Eye_Contact_Ratio", "Vocal_Pitch_Var", "Speech_Rate", "Pause_Count",
    "Laughter_Duration_Ratio", "overall__lead_speaking_time_s",
    "Speaker_Gender", "Judge_Male_Rate",
]
ENCODINGS = {
    "AVG": "Joke_Attempt_Frequency_ave",
    "BASE": "Joke_Attempt_Frequency",
    "SECOND": "Joke_Attempt_Frequency_second",
}
LABELS = {
    "money_log": "Log funding",
    "Joke_Attempt_Frequency_ave": "Humor frequency",
    "S_ICL": "Warmth cues",
    "Pocket_Hands_Ratio": "Hands-in-pockets ratio",
    "Hand Gesture Frequency": "Hand-gesture frequency",
    "Body_Openness": "Body openness",
    "Eye_Contact_Ratio": "Eye-contact proxy",
    "Vocal_Pitch_Var": "Vocal-pitch variation",
    "Speech_Rate": "Speech rate",
    "Pause_Count": "Pauses per 100 words",
    "Laughter_Duration_Ratio": "Detected-laughter ratio",
    "overall__lead_speaking_time_s": "Effective speaking time (s)",
    "Speaker_Gender": "Female lead speaker",
    "Judge_Male_Rate": "Male share of judges",
}


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", float_format="%.17g")


def prepare_sample(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the common complete-case and transcript-quality selection rule."""
    required = [
        "Money_Usd", "Joke_Attempt", "Joke_Attempt_second",
        "Smile_Duration_Ratio", "inclusive_collective_language", *CONTROLS,
    ]
    missing = set(required + ["video_id", "text_valid", "api_status"]) - set(data)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    if not data.video_id.is_unique:
        raise ValueError("Each input row must have a unique video_id.")
    numeric = data[required].apply(pd.to_numeric, errors="raise")
    quality = data.text_valid.astype(str).str.lower().eq("true")
    eligible = (np.isfinite(numeric).all(axis=1) & quality
                & data.api_status.eq("success")
                & numeric.overall__lead_speaking_time_s.gt(0)
                & numeric.Money_Usd.ge(0))
    sample = data.loc[eligible].copy()
    sample[required] = numeric.loc[eligible]
    if len(sample) <= len(CONTROLS) + 5:
        raise ValueError("Insufficient complete observations for the full model.")
    sample["overall__lead_speaking_minutes"] = sample.overall__lead_speaking_time_s / 60
    for count, frequency in [
        ("Joke_Attempt", "Joke_Attempt_Frequency"),
        ("Joke_Attempt_second", "Joke_Attempt_Frequency_second"),
    ]:
        sample[frequency] = sample[count] / sample.overall__lead_speaking_minutes
    sample["Joke_Attempt_Frequency_ave"] = (
        sample.Joke_Attempt_Frequency + sample.Joke_Attempt_Frequency_second
    ) / 2
    sample["money_log"] = np.log1p(sample.Money_Usd)
    for source, target in [("Smile_Duration_Ratio", "z_S"),
                           ("inclusive_collective_language", "z_ICL")]:
        scale = sample[source].std(ddof=1)
        if scale <= 0:
            raise ValueError(f"Cannot standardize constant variable {source}.")
        sample[target] = (sample[source] - sample[source].mean()) / scale
    sample["S_ICL"] = (sample.z_S + sample.z_ICL) / 2
    return sample


def fit_model(sample, encoding, stage, controlled):
    h = sample[ENCODINGS[encoding]]
    hc = h - h.mean()
    wc = sample.S_ICL - sample.S_ICL.mean()
    design = pd.DataFrame({"Intercept": 1., "humor_c": hc, "humor_c2": hc**2})
    if stage in ("main_effects", "interaction"):
        design["S_ICL_c"] = wc
    if stage == "interaction":
        design["humor_x_S_ICL"] = hc * wc
    if controlled:
        design = pd.concat([design, sample[CONTROLS]], axis=1)
    fit = sm.OLS(sample.money_log, design).fit(use_t=True)
    identifier = f"{encoding}_{'NONE' if stage == 'baseline' else 'S_ICL'}_{stage}_{'C' if controlled else 'NC'}"
    return identifier, fit, design


def coefficient_rows(identifier, fit, table, model):
    intervals = fit.conf_int(alpha=.05)
    return [
        {"table": table, "model": model, "model_id": identifier, "term": term,
         "b": float(fit.params[term]), "se": float(fit.bse[term]),
         "t": float(fit.tvalues[term]), "p": float(fit.pvalues[term]),
         "ci_low": float(intervals.loc[term, 0]),
         "ci_high": float(intervals.loc[term, 1]), "df_resid": int(fit.df_resid)}
        for term in fit.params.index
    ]


def covariance_sensitivity(sample: pd.DataFrame, fit) -> list[dict]:
    """Evaluate the primary interaction under seven covariance estimators."""
    interaction_index = fit.params.index.get_loc("humor_x_S_ICL")
    estimate = float(fit.params.iloc[interaction_index])
    rows = []
    company_labels = sample["Company Name"].astype(str).str.strip()
    event_labels = sample["Alum & Event"].fillna("Unknown").astype(str)
    for name in ["Classical OLS", "HC0", "HC1", "HC2", "HC3",
                 "Company CR1", "Legacy event CR1"]:
        groups = None
        if name == "Classical OLS":
            inference = fit
            degrees = int(fit.df_resid)
        elif name.startswith("HC"):
            inference = fit.get_robustcov_results(cov_type=name, use_t=True)
            degrees = int(fit.df_resid)
        else:
            labels = company_labels if name == "Company CR1" else event_labels
            codes, unique = pd.factorize(labels, sort=True)
            groups = len(unique)
            degrees = groups - 1
            inference = fit.get_robustcov_results(
                cov_type="cluster", groups=codes, use_correction=True,
                df_correction=True, use_t=True,
            )
        se = float(np.asarray(inference.bse)[interaction_index])
        critical = float(stats.t.ppf(.975, degrees))
        row = {"covariance": name, "n": len(sample), "b": estimate, "se": se,
               "df": degrees, "p": float(2*stats.t.sf(abs(estimate/se), degrees)),
               "ci_low": estimate-critical*se, "ci_high": estimate+critical*se,
               "groups": groups}
        rows.append(row)
    return rows


def fieller(fit, numerator, denominator):
    """95% Fieller interval for a ratio of correlated coefficient estimates."""
    n = float(numerator @ fit.params)
    d = float(denominator @ fit.params)
    covariance = fit.cov_params().to_numpy()
    critical2 = stats.t.ppf(.975, fit.df_resid)**2
    a = d*d - critical2 * (denominator @ covariance @ denominator)
    b = -2*n*d + 2*critical2 * (numerator @ covariance @ denominator)
    c = n*n - critical2 * (numerator @ covariance @ numerator)
    discriminant = b*b - 4*a*c
    estimate = n/d
    if a > 0 and discriminant >= 0:
        roots = sorted([(-b-np.sqrt(discriminant))/(2*a),
                        (-b+np.sqrt(discriminant))/(2*a)])
        return {"estimate": estimate, "type": "bounded",
                "ci_low": float(roots[0]), "ci_high": float(roots[1])}
    return {"estimate": estimate, "type": "unbounded_or_disjoint",
            "quadratic_a": float(a), "quadratic_b": float(b), "quadratic_c": float(c)}


def summarize_figure(sample, fit, out):
    h = sample.Joke_Attempt_Frequency_ave
    h_mean, w_mean = float(h.mean()), float(sample.S_ICL.mean())
    levels = {"P25": float(sample.S_ICL.quantile(.25)),
              "P75": float(sample.S_ICL.quantile(.75))}
    grid = np.linspace(h.min(), h.max(), 601)
    parameters = fit.params
    n_terms = len(parameters)
    h_index = parameters.index.get_loc("humor_c")
    h2_index = parameters.index.get_loc("humor_c2")
    interaction_index = parameters.index.get_loc("humor_x_S_ICL")

    def design(humor, warmth):
        humor = np.atleast_1d(humor)
        centered = humor - h_mean
        x = pd.DataFrame({"Intercept": np.ones(len(humor)), "humor_c": centered,
                          "humor_c2": centered**2, "S_ICL_c": warmth-w_mean,
                          "humor_x_S_ICL": centered*(warmth-w_mean)})
        for column in CONTROLS:
            x[column] = sample[column].mean()
        return x[parameters.index]

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.45, 4.8))
    colors = {"P25": "#555555", "P75": "#1b5b83"}
    labels = {"P25": "Low warmth cues (25th percentile)",
              "P75": "High warmth cues (75th percentile)"}
    predictions, vertices = [], {}
    critical = stats.t.ppf(.975, fit.df_resid)
    for key, warmth in levels.items():
        x = design(grid, warmth)
        estimate = x.to_numpy() @ parameters.to_numpy()
        se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", x, fit.cov_params(), x), 0))
        low, high = estimate-critical*se, estimate+critical*se
        for i in range(len(grid)):
            predictions.append({"warmth_level": key, "warmth_value": warmth,
                                "humor_frequency": float(grid[i]),
                                "predicted_log_funding": float(estimate[i]),
                                "ci_low": float(low[i]), "ci_high": float(high[i])})
        vertex = h_mean - (parameters.humor_c + parameters.humor_x_S_ICL*(warmth-w_mean)) / (2*parameters.humor_c2)
        peak_y = float(fit.predict(design(vertex, warmth)).iloc[0])
        numerator, denominator = np.zeros(n_terms), np.zeros(n_terms)
        numerator[h_index], numerator[interaction_index] = 1, warmth-w_mean
        denominator[h2_index] = -2
        vertex_ci = fieller(fit, numerator, denominator)
        vertex_ci["estimate"] += h_mean
        if vertex_ci["type"] == "bounded":
            vertex_ci["ci_low"] += h_mean
            vertex_ci["ci_high"] += h_mean
        vertices[key] = {"warmth": warmth, "humor_frequency": float(vertex),
                         "predicted_log_funding": peak_y, "fieller_95ci": vertex_ci}
        ax.fill_between(grid, low, high, color=colors[key], alpha=.09, lw=0)
        ax.plot(grid, estimate, color=colors[key], lw=1.8, label=labels[key])
        ax.scatter([vertex], [peak_y], s=25, color=colors[key], zorder=6)
        ax.annotate(f"{vertex:.3f}", (vertex, peak_y),
                    xytext=(-45, 20) if key == "P25" else (35, 20),
                    textcoords="offset points", color=colors[key], fontsize=10,
                    arrowprops={"arrowstyle": "-", "color": colors[key], "lw": .65})
    difference = levels["P75"] - levels["P25"]
    numerator, denominator = np.zeros(n_terms), np.zeros(n_terms)
    numerator[interaction_index], denominator[h2_index] = difference, -2
    peak_shift = fieller(fit, numerator, denominator)
    contrast = np.zeros(n_terms)
    contrast[interaction_index] = difference * .1
    estimate = float(contrast @ parameters)
    se = float(np.sqrt(contrast @ fit.cov_params() @ contrast))
    slope_difference = {"b": estimate, "se": se,
                        "p": float(2*stats.t.sf(abs(estimate/se), fit.df_resid)),
                        "ci_low": estimate-critical*se, "ci_high": estimate+critical*se}
    ax.plot(h, np.full(len(h), .012), "|", transform=ax.get_xaxis_transform(),
            ms=4, alpha=.25, color="#555555")
    ax.set(xlim=(grid.min(), grid.max()), xlabel="Humor attempts per effective speaking minute",
           ylabel="Predicted log funding, ln(1 + US dollars)")
    ax.grid(axis="y", color="#dddddd", lw=.45)
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    fig.tight_layout(pad=1.15)
    fig.savefig(out / "figure1.png", dpi=300, facecolor="white")
    plt.close(fig)
    write_csv(pd.DataFrame(predictions), out / "figure1_predictions.csv")
    duration = float(sample.overall__lead_speaking_minutes.mean())
    return {"levels": levels, "vertices": vertices,
            "peak_shift_high_minus_low": peak_shift,
            "slope_difference_per_0_1_humor": slope_difference,
            "mean_effective_speaking_minutes": duration,
            "peak_count_equivalents": {key: value["humor_frequency"] * duration
                                       for key, value in vertices.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/analysis/input_428.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    sample = prepare_sample(data)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(sample, args.output / "analysis_sample_371.csv")

    variables = ["money_log", "Joke_Attempt_Frequency_ave", "S_ICL", *CONTROLS]
    descriptives = pd.DataFrame([
        {"number": i+1, "variable": column, "label": LABELS[column],
         "n": int(sample[column].count()), "mean": float(sample[column].mean()),
         "sd": float(sample[column].std(ddof=1))}
        for i, column in enumerate(variables)
    ])
    write_csv(descriptives, args.output / "table1_descriptives.csv")
    correlation = sample[variables].corr(method="pearson")
    write_csv(correlation.rename_axis("variable").reset_index(),
              args.output / "table1_correlations.csv")

    coefficients, statistics, primary_fit, primary_design = [], [], None, None
    specifications = [(2, f"Model {i+1}", "AVG", stage, controlled)
                      for i, (controlled, stage) in enumerate(
                          (controlled, stage) for controlled in (False, True)
                          for stage in ("baseline", "main_effects", "interaction"))]
    specifications += [(3, f"Version {version}, {'with' if controlled else 'no'} controls",
                        encoding, "interaction", controlled)
                       for version, encoding in (("A", "BASE"), ("B", "SECOND"))
                       for controlled in (False, True)]
    for table, label, encoding, stage, controlled in specifications:
        identifier, fit, design = fit_model(sample, encoding, stage, controlled)
        coefficients.extend(coefficient_rows(identifier, fit, table, label))
        statistics.append({"table": table, "model": label, "model_id": identifier,
                           "encoding": encoding, "stage": stage, "controls": controlled,
                           "n": int(fit.nobs), "k": len(fit.params),
                           "df_resid": int(fit.df_resid), "r2": float(fit.rsquared),
                           "adj_r2": float(fit.rsquared_adj),
                           "f": float(fit.fvalue), "f_p": float(fit.f_pvalue)})
        if table == 2 and label == "Model 6":
            primary_fit, primary_design = fit, design
    coefficients = pd.DataFrame(coefficients)
    for table in (2, 3):
        write_csv(coefficients[coefficients.table.eq(table)],
                  args.output / f"table{table}_coefficients.csv")
    write_csv(pd.DataFrame(statistics), args.output / "model_statistics.csv")
    sensitivity = covariance_sensitivity(sample, primary_fit)
    write_csv(pd.DataFrame(sensitivity), args.output / "covariance_sensitivity.csv")
    vif = [{"term": term, "vif": float(variance_inflation_factor(primary_design.to_numpy(), i))}
           for i, term in enumerate(primary_design.columns) if term != "Intercept"]
    write_csv(pd.DataFrame(vif), args.output / "vif.csv")
    figure = summarize_figure(sample, primary_fit, args.output)
    primary_rows = coefficients[coefficients.model_id.eq("AVG_S_ICL_interaction_C")]
    primary = {row["term"]: {key: row[key] for key in ("b", "se", "p", "ci_low", "ci_high")}
               for row in primary_rows.to_dict("records")}
    original_columns = ["Money_Usd", "Joke_Attempt", "Joke_Attempt_second",
                        "Smile_Duration_Ratio", *CONTROLS]
    original_numeric = data[original_columns].apply(pd.to_numeric, errors="raise")
    original_complete = (np.isfinite(original_numeric).all(axis=1)
                         & original_numeric.overall__lead_speaking_time_s.gt(0)
                         & original_numeric.Money_Usd.ge(0))
    summary = {
        "assembled_n": len(data), "analysis_n": len(sample), "models": len(statistics),
        "original_complete_n": int(original_complete.sum()),
        "transcript_valid_n": int(data.text_valid.astype(str).str.lower().eq("true").sum()),
        "company_n": int(sample["Company Name"].nunique()),
        "company_covariance_groups": int(sample["Company Name"].astype(str).str.strip().nunique()),
        "event_covariance_groups": int(sample["Alum & Event"].fillna("Unknown").astype(str).nunique()),
        "missing_event_labels": int(sample["Alum & Event"].isna().sum()),
        "controls": CONTROLS, "primary_model_id": "AVG_S_ICL_interaction_C",
        "primary_coefficients": primary,
        "humor_coding_pearson_r": float(sample[["Joke_Attempt_Frequency", "Joke_Attempt_Frequency_second"]].corr().iloc[0, 1]),
        "smile_icl_pearson_r": float(sample[["Smile_Duration_Ratio", "inclusive_collective_language"]].corr().iloc[0, 1]),
        "warmth_sd": float(sample.S_ICL.std(ddof=1)),
        "smile_mean": float(sample.Smile_Duration_Ratio.mean()),
        "smile_sd": float(sample.Smile_Duration_Ratio.std(ddof=1)),
        "funding_mean_usd": float(sample.Money_Usd.mean()),
        "funding_median_usd": float(sample.Money_Usd.median()),
        "vif_min": min(row["vif"] for row in vif),
        "vif_max": max(row["vif"] for row in vif),
        "figure1": figure,
        "covariance_sensitivity": sensitivity,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"assembled_n": len(data), "analysis_n": len(sample),
                      "models": len(statistics), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
