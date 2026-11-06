
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm


INPUT_DIR = r"\result-PVpot"
OUTPUT_DIR = r"result-PVpot CV"

SCENARIOS = ["SSP126", "SSP245", "SSP585"]

HISTORICAL_PERIOD = ("2015", "2025")
FUTURE_PERIOD = ("2026", "2100")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_grid_data(scenario):

    file_path = os.path.join(
        INPUT_DIR,
        f"PVpot_{scenario}-2015-2100.csv"
    )

    try:
        df = pd.read_csv(file_path)

        required_cols = ["lat", "lon"]

        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")

        df = df.loc[
            :,
            ~df.columns.str.startswith("Unnamed")
        ].copy()

        if "geometry" not in df.columns:
            df["geometry"] = [
                f"POINT ({lon} {lat})"
                for lon, lat in zip(df["lon"], df["lat"])
            ]

        date_cols = [
            col for col in df.columns
            if col not in ["lat", "lon", "geometry"]
            and not pd.isna(
                pd.to_datetime(col, format="%Y-%m-%d", errors="coerce")
            )
        ]

        if not date_cols:
            raise ValueError("No valid date columns found.")

        expected_days = 31411

        if len(date_cols) != expected_days:
            print(
                f"Warning: {len(date_cols)} date columns found; "
                f"{expected_days} expected."
            )

        data = df.set_index(
            ["lat", "lon", "geometry"]
        )[date_cols].T

        data.index = pd.to_datetime(
            data.index,
            format="%Y-%m-%d"
        )

        return data

    except Exception as e:
        print(f"Failed to load {scenario}: {e}")
        return None


def sens_slope_mk_test(data):

    data = np.asarray(data, dtype=float)

    valid_idx = np.flatnonzero(np.isfinite(data))
    valid_data = data[valid_idx]

    n = len(valid_data)

    if n < 2:
        return np.nan, np.nan, np.nan

    slopes = []

    for i in range(n - 1):
        for j in range(i + 1, n):
            slope = (
                (valid_data[j] - valid_data[i])
                / (valid_idx[j] - valid_idx[i])
            )
            slopes.append(slope)

    sen_slope = np.median(slopes) * 10

    s = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(valid_data[j] - valid_data[i])

    _, tie_counts = np.unique(
        valid_data,
        return_counts=True
    )

    var_s = (
        n * (n - 1) * (2 * n + 5)
        - np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5))
    ) / 18

    if var_s <= 0:
        return sen_slope, 0.0, 1.0

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p = 2 * (1 - norm.cdf(abs(z)))

    return sen_slope, z, p


def analyze_grid_point(data, grid_point):

    yield_data = data[grid_point]

    annual_energy = yield_data.resample("YE").sum()

    hist_energy = annual_energy.loc[
        HISTORICAL_PERIOD[0]:HISTORICAL_PERIOD[1]
    ]

    future_energy = annual_energy.loc[
        FUTURE_PERIOD[0]:FUTURE_PERIOD[1]
    ]

    hist_mean = hist_energy.mean()
    hist_std = hist_energy.std()

    slope_yield, z_yield, p_yield = sens_slope_mk_test(
        future_energy.values
    )

    high_threshold = hist_energy.quantile(0.90)
    low_threshold = hist_energy.quantile(0.10)

    future_daily = yield_data.loc[
        FUTURE_PERIOD[0]:FUTURE_PERIOD[1]
    ]

    extreme_high = (
        (future_daily > high_threshold)
        .astype(int)
        .resample("YE")
        .sum()
    )

    extreme_low = (
        (future_daily < low_threshold)
        .astype(int)
        .resample("YE")
        .sum()
    )

    slope_high, z_high, p_high = sens_slope_mk_test(
        extreme_high.values
    )

    slope_low, z_low, p_low = sens_slope_mk_test(
        extreme_low.values
    )

    cv_hist = (
        hist_std / hist_mean
        if hist_mean > 1e-6 else np.nan
    )

    cv_overall = (
        future_energy.std() / hist_mean
        if hist_mean > 1e-6 else np.nan
    )

    annual_data = future_daily.groupby(
        future_daily.index.year
    )

    cv_series = annual_data.apply(
        lambda x: (
            x.std() / x.mean()
            if x.mean() > 1e-6 else np.nan
        )
    )

    slope_cv, z_cv, p_cv = sens_slope_mk_test(
        cv_series.values
    )

    lat, lon, geometry = grid_point

    return {
        "lat": lat,
        "lon": lon,
        "geometry": geometry,

        "Historical_mean": round(hist_mean, 3),
        "Future_mean": round(future_energy.mean(), 3),

        "PVpot_Sen_slope_(%/decade)": (
            round(slope_yield / hist_mean * 100, 2)
            if hist_mean > 1e-6 else np.nan
        ),
        "PVpot_MK_p_value": round(p_yield, 4),
        "PVpot_trend": (
            "Increasing"
            if p_yield < 0.05 and slope_yield > 0
            else "Decreasing"
            if p_yield < 0.05 and slope_yield < 0
            else "No significant trend"
        ),

        "Extreme_high_days_per_year": round(
            extreme_high.mean(), 1
        ),
        "Extreme_high_trend_(days/decade)": round(
            slope_high, 1
        ),
        "Extreme_high_MK_p_value": round(p_high, 4),

        "Extreme_low_days_per_year": round(
            extreme_low.mean(), 1
        ),
        "Extreme_low_trend_(days/decade)": round(
            slope_low, 1
        ),
        "Extreme_low_MK_p_value": round(p_low, 4),

        "Historical_CV": round(cv_hist, 3),
        "Future_interannual_CV": round(cv_overall, 3),

        "CV_trend_slope_(%/decade)": round(
            slope_cv * 100, 4
        ),
        "CV_trend_p_value": round(p_cv, 4),
        "CV_trend": (
            "Increasing"
            if p_cv < 0.05 and slope_cv > 0
            else "Decreasing"
            if p_cv < 0.05 and slope_cv < 0
            else "No significant trend"
        )
    }


def process_scenario(scenario):

    print(f"\nProcessing {scenario}...")

    data = load_grid_data(scenario)

    if data is None:
        print(f"Skipping {scenario}.")
        return

    results = []

    for grid_point in tqdm(
        data.columns,
        desc=f"Analyzing {scenario}"
    ):

        try:
            result = analyze_grid_point(
                data,
                grid_point
            )

            results.append(result)

        except Exception as e:
            print(
                f"Error at grid point {grid_point}: {e}"
            )

    if not results:
        print(f"No valid results for {scenario}.")
        return

    result_df = pd.DataFrame(results)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{scenario}_grid_analysis.csv"
    )

    try:
        result_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"Results saved: {output_path}")

    except Exception as e:
        print(f"Failed to save {scenario}: {e}")


if __name__ == "__main__":

    for scenario in SCENARIOS:
        process_scenario(scenario)

    print("\nAll scenarios completed.")
