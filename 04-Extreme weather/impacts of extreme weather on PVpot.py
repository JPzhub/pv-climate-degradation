# -*- coding: utf-8 -*-

import os
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter


warnings.filterwarnings("ignore", category=RuntimeWarning)


PV_DATA_DIR = r"result-PVpot"
CLIMATE_DATA_DIR = r"NEX-GDDP-CMIP6\sfcWind-csv-daily"
OUTPUT_DIR = r"impacts of extreme weather on PVpot"

SCENARIOS = ["SSP126", "SSP245", "SSP585"]

HIST_START = 2015
HIST_END = 2025

FUTURE_PERIODS = {
    "2026-2050": (2026, 2050),
    "2051-2075": (2051, 2075),
    "2076-2100": (2076, 2100),
    "2026-2100": (2026, 2100)
}

QUANTILE_HIGH = 0.90
QUANTILE_LOW = 0.10

MIN_VALID_FUTURE_YEARS = 1
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXTREME_SCENARIOS = {
    "Original": "All days retained",
    "Exclude_High_Temperature":
        "Days above the historical 90th temperature percentile excluded",
    "Exclude_Low_Temperature":
        "Days below the historical 10th temperature percentile excluded",
    "Exclude_High_Radiation":
        "Days above the historical 90th radiation percentile excluded",
    "Exclude_Low_Radiation":
        "Days below the historical 10th radiation percentile excluded",
    "Exclude_All_Extremes":
        "Days outside historical temperature and radiation percentile ranges excluded",
    "Exclude_Both_Temperature_Extremes":
        "Days outside the historical temperature percentile range excluded",
    "Exclude_Both_Radiation_Extremes":
        "Days outside the historical radiation percentile range excluded"
}

def parse_date_columns(columns):
    """Identify columns that can be interpreted as dates."""

    date_columns = []

    for col in columns:
        parsed_date = pd.to_datetime(
            str(col),
            format="mixed",
            errors="coerce"
        )

        if not pd.isna(parsed_date):
            date_columns.append((col, parsed_date))

    return date_columns


def load_analysis_data(scenario, variable):

    if variable == "PVpot":
        file_path = os.path.join(
            PV_DATA_DIR,
            f"PVpot_{scenario}-2015-2100.csv"
        )
    else:
        file_path = os.path.join(
            CLIMATE_DATA_DIR,
            f"{scenario}-2015-2100-{variable}-日.csv"
        )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_csv(file_path, encoding="gbk")

    df = df.loc[
        :,
        ~df.columns.astype(str).str.startswith("Unnamed:")
    ]

    for col in ["lat", "lon"]:
        if col not in df.columns:
            raise ValueError(
                f"Missing required column '{col}' in {file_path}"
            )

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    if df[["lat", "lon"]].isna().any().any():
        raise ValueError(
            f"Invalid latitude or longitude values in {file_path}"
        )

    if "geometry" not in df.columns:
        df["geometry"] = [
            f"POINT ({lon} {lat})"
            for lat, lon in zip(df["lat"], df["lon"])
        ]

    date_info = parse_date_columns(df.columns)

    if not date_info:
        raise ValueError(
            f"No valid date columns found in {file_path}"
        )

    original_date_columns = [item[0] for item in date_info]
    parsed_dates = [item[1] for item in date_info]

    if len(set(parsed_dates)) != len(parsed_dates):
        raise ValueError(f"Duplicate dates found in {file_path}")

    index_columns = ["lat", "lon", "geometry"]

    data = df[index_columns + original_date_columns].copy()
    data = data.set_index(index_columns).T

    data.index = pd.DatetimeIndex(parsed_dates)
    data.index.name = "date"

    data.columns = pd.MultiIndex.from_tuples(
        data.columns,
        names=["lat", "lon", "geometry"]
    )

    data = data.sort_index()
    data = data.apply(pd.to_numeric, errors="coerce")

    if variable == "tas":
        data = data - 273.15

    return data

def prepare_analysis_data(scenario):
    print(f"\nLoading data for {scenario}...")

    pvpot = load_analysis_data(scenario, "PVpot")
    temperature = load_analysis_data(scenario, "tas")
    radiation = load_analysis_data(scenario, "rsds")

    common_columns = (
        pvpot.columns
        .intersection(temperature.columns)
        .intersection(radiation.columns)
    )

    if len(common_columns) == 0:
        raise ValueError(f"No common grid cells found for {scenario}")

    pvpot = pvpot.loc[:, common_columns]
    temperature = temperature.loc[:, common_columns]
    radiation = radiation.loc[:, common_columns]

    common_dates = (
        pvpot.index
        .intersection(temperature.index)
        .intersection(radiation.index)
    )

    pvpot = pvpot.loc[common_dates]
    temperature = temperature.loc[common_dates]
    radiation = radiation.loc[common_dates]

    if pvpot.empty:
        raise ValueError(f"No overlapping dates found for {scenario}")

    hist_mask = (
        (pvpot.index.year >= HIST_START) &
        (pvpot.index.year <= HIST_END)
    )

    hist_pvpot = pvpot.loc[hist_mask]
    hist_temperature = temperature.loc[hist_mask]
    hist_radiation = radiation.loc[hist_mask]

    if hist_pvpot.empty:
        raise ValueError(f"No historical data found for {scenario}")

    historical_baseline = hist_pvpot.mean(axis=0)

    temperature_q90 = hist_temperature.quantile(
        QUANTILE_HIGH,
        axis=0
    )
    temperature_q10 = hist_temperature.quantile(
        QUANTILE_LOW,
        axis=0
    )
    radiation_q90 = hist_radiation.quantile(
        QUANTILE_HIGH,
        axis=0
    )
    radiation_q10 = hist_radiation.quantile(
        QUANTILE_LOW,
        axis=0
    )

    historical_annual_energy = (
        hist_pvpot
        .resample("YE")
        .sum(min_count=1)
    )

    historical_annual_baseline = historical_annual_energy.mean(axis=0)

    return {
        "pvpot": pvpot,
        "temperature": temperature,
        "radiation": radiation,
        "historical_pvpot": hist_pvpot,
        "historical_baseline": historical_baseline,
        "historical_annual_baseline": historical_annual_baseline,
        "temperature_q90": temperature_q90,
        "temperature_q10": temperature_q10,
        "radiation_q90": radiation_q90,
        "radiation_q10": radiation_q10,
        "common_columns": common_columns
    }

def build_retention_masks(
    temperature,
    radiation,
    temperature_q90,
    temperature_q10,
    radiation_q90,
    radiation_q10
):
    temp_valid = temperature.notna()
    rad_valid = radiation.notna()

    temp_high = temperature.gt(temperature_q90, axis=1)
    temp_low = temperature.lt(temperature_q10, axis=1)

    rad_high = radiation.gt(radiation_q90, axis=1)
    rad_low = radiation.lt(radiation_q10, axis=1)

    masks = {
        "Original": pd.DataFrame(
            True,
            index=temperature.index,
            columns=temperature.columns
        ),

        "Exclude_High_Temperature":
            temp_valid & ~temp_high,

        "Exclude_Low_Temperature":
            temp_valid & ~temp_low,

        "Exclude_High_Radiation":
            rad_valid & ~rad_high,

        "Exclude_Low_Radiation":
            rad_valid & ~rad_low,

        "Exclude_All_Extremes":
            temp_valid & rad_valid &
            ~(temp_high | temp_low | rad_high | rad_low),

        "Exclude_Both_Temperature_Extremes":
            temp_valid & ~(temp_high | temp_low),

        "Exclude_Both_Radiation_Extremes":
            rad_valid & ~(rad_high | rad_low)
    }

    return masks
def calculate_daily_growth(
    future_pvpot,
    retention_mask,
    historical_baseline
):
    retained_pvpot = future_pvpot.where(retention_mask)
    future_mean = retained_pvpot.mean(axis=0)

    growth_rate = (
        (future_mean - historical_baseline) /
        historical_baseline.replace(0, np.nan)
    ) * 100

    return growth_rate
def calculate_annual_energy_growth(
    future_pvpot,
    retention_mask,
    historical_annual_baseline,
    min_valid_years=MIN_VALID_FUTURE_YEARS
):
    retained_pvpot = future_pvpot.where(retention_mask)

    annual_energy = (
        retained_pvpot
        .resample("YE")
        .sum(min_count=1)
    )

    valid_year_count = annual_energy.count(axis=0)
    future_mean_annual_energy = annual_energy.mean(axis=0)

    future_mean_annual_energy = future_mean_annual_energy.where(
        valid_year_count >= min_valid_years
    )

    growth_rate = (
        (future_mean_annual_energy - historical_annual_baseline) /
        historical_annual_baseline.replace(0, np.nan)
    ) * 100

    return growth_rate, future_mean_annual_energy, valid_year_count
def create_result_records(
    growth_rate,
    period,
    scenario_name,
    metric_name,
    common_columns,
    future_mean=None,
    valid_year_count=None
):
    records = []

    for grid_point in common_columns:
        lat, lon, geometry = grid_point

        record = {
            "Latitude": lat,
            "Longitude": lon,
            "Geometry": geometry,
            "Period": period,
            "Extreme_Weather_Scenario": scenario_name,
            "Metric": metric_name,
            "Growth_Rate_Percent": growth_rate.get(grid_point, np.nan)
        }

        if future_mean is not None:
            record["Future_Mean_Value"] = future_mean.get(
                grid_point,
                np.nan
            )

        if valid_year_count is not None:
            record["Valid_Future_Year_Count"] = valid_year_count.get(
                grid_point,
                np.nan
            )

        records.append(record)

    return records
def analyze_scenario(scenario):
    data = prepare_analysis_data(scenario)
    pvpot = data["pvpot"]
    temperature = data["temperature"]
    radiation = data["radiation"]
    common_columns = data["common_columns"]

    masks = build_retention_masks(
        temperature,
        radiation,
        data["temperature_q90"],
        data["temperature_q10"],
        data["radiation_q90"],
        data["radiation_q10"]
    )

    daily_records = []
    annual_records = []

    for period, (start_year, end_year) in FUTURE_PERIODS.items():

        print(f"\nAnalyzing {scenario}: {period}")

        period_mask = (
            (pvpot.index.year >= start_year) &
            (pvpot.index.year <= end_year)
        )

        future_pvpot = pvpot.loc[period_mask]

        if future_pvpot.empty:
            print(f"No data available for {period}. Skipping.")
            continue

        for scenario_name in tqdm(
            EXTREME_SCENARIOS.keys(),
            desc=f"{scenario} {period}"
        ):

            retention_mask = masks[scenario_name].loc[
                future_pvpot.index
            ]

            daily_growth = calculate_daily_growth(
                future_pvpot,
                retention_mask,
                data["historical_baseline"]
            )

            daily_records.extend(
                create_result_records(
                    growth_rate=daily_growth,
                    period=period,
                    scenario_name=scenario_name,
                    metric_name="Daily_PVpot_Growth",
                    common_columns=common_columns
                )
            )

            (
                annual_growth,
                future_mean_annual_energy,
                valid_year_count
            ) = calculate_annual_energy_growth(
                future_pvpot,
                retention_mask,
                data["historical_annual_baseline"]
            )

            annual_records.extend(
                create_result_records(
                    growth_rate=annual_growth,
                    period=period,
                    scenario_name=scenario_name,
                    metric_name="Annual_Energy_Growth",
                    common_columns=common_columns,
                    future_mean=future_mean_annual_energy,
                    valid_year_count=valid_year_count
                )
            )

    daily_results = pd.DataFrame(daily_records)
    annual_results = pd.DataFrame(annual_records)

    return daily_results, annual_results
def calculate_area_weighted_summary(results, metric_name):
    if results.empty:
        return pd.DataFrame()

    required_columns = {
        "Latitude",
        "Period",
        "Extreme_Weather_Scenario",
        "Growth_Rate_Percent"
    }

    missing_columns = required_columns.difference(results.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns for area-weighted summary: {missing_columns}"
        )

    summary_records = []

    group_columns = ["Period", "Extreme_Weather_Scenario"]

    for (period, scenario_name), group in results.groupby(
        group_columns,
        sort=False
    ):

        values = pd.to_numeric(
            group["Growth_Rate_Percent"],
            errors="coerce"
        ).to_numpy(dtype=float)

        latitudes = pd.to_numeric(
            group["Latitude"],
            errors="coerce"
        ).to_numpy(dtype=float)

        weights = np.cos(np.deg2rad(latitudes))

        valid = (
            np.isfinite(values) &
            np.isfinite(weights) &
            (weights > 0)
        )

        total_grid_count = len(group)
        valid_grid_count = int(valid.sum())

        all_weights_valid = (
            np.isfinite(weights) &
            (weights > 0)
        )

        total_area_weight = weights[all_weights_valid].sum()
        valid_area_weight = weights[valid].sum()

        if valid_grid_count == 0 or valid_area_weight <= 0:
            weighted_mean = np.nan
            weighted_std = np.nan
            area_coverage = 0.0
        else:
            valid_values = values[valid]
            valid_weights = weights[valid]

            weighted_mean = np.average(
                valid_values,
                weights=valid_weights
            )

            weighted_variance = np.average(
                (valid_values - weighted_mean) ** 2,
                weights=valid_weights
            )

            weighted_std = np.sqrt(weighted_variance)

            area_coverage = (
                valid_area_weight / total_area_weight * 100
                if total_area_weight > 0
                else np.nan
            )

        summary_records.append({
            "Metric": metric_name,
            "Period": period,
            "Extreme_Weather_Scenario": scenario_name,
            "Area_Weighted_Mean_Growth_Rate": weighted_mean,
            "Area_Weighted_Std_Growth_Rate": weighted_std,
            "Valid_Grid_Count": valid_grid_count,
            "Total_Grid_Count": total_grid_count,
            "Area_Coverage_Percent": area_coverage
        })

    return pd.DataFrame(summary_records)
def create_period_pivot(results, period):
    if results.empty:
        return pd.DataFrame()

    period_data = results.loc[results["Period"] == period]

    if period_data.empty:
        return pd.DataFrame()

    pivot = period_data.pivot_table(
        index=["Latitude", "Longitude", "Geometry"],
        columns="Extreme_Weather_Scenario",
        values="Growth_Rate_Percent",
        aggfunc="first"
    )

    pivot = pivot.reset_index()
    pivot.columns.name = None

    return pivot


def create_full_period_table(results):

    return create_period_pivot(results, "2026-2100")

def write_dataframe(writer, dataframe, sheet_name):
    if dataframe is None or dataframe.empty:
        pd.DataFrame(
            {"Message": ["No data available"]}
        ).to_excel(
            writer,
            sheet_name=sheet_name,
            index=False
        )
    else:
        dataframe.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False
        )

def format_excel(file_path):
    workbook = load_workbook(file_path)

    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        for column_cells in worksheet.columns:
            max_length = 0
            column_index = column_cells[0].column
            column_letter = get_column_letter(column_index)

            for cell in column_cells:
                if cell.value is not None:
                    max_length = max(
                        max_length,
                        len(str(cell.value))
                    )

            worksheet.column_dimensions[column_letter].width = min(
                max_length + 2,
                40
            )

    workbook.save(file_path)

def export_scenario_results(
    scenario,
    daily_results,
    annual_results
):
    output_file = os.path.join(
        OUTPUT_DIR,
        f"{scenario}_Extreme_Weather_PVpot_Analysis.xlsx"
    )

    daily_summary = calculate_area_weighted_summary(
        daily_results,
        "Daily_PVpot_Growth"
    )

    annual_summary = calculate_area_weighted_summary(
        annual_results,
        "Annual_Energy_Growth"
    )

    national_summary = pd.concat(
        [daily_summary, annual_summary],
        ignore_index=True
    )

    daily_period_tables = {
        period: create_period_pivot(daily_results, period)
        for period in FUTURE_PERIODS
    }

    annual_period_tables = {
        period: create_period_pivot(annual_results, period)
        for period in FUTURE_PERIODS
    }

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

        write_dataframe(
            writer,
            daily_period_tables["2026-2050"],
            "Daily_2026_2050"
        )
        write_dataframe(
            writer,
            daily_period_tables["2051-2075"],
            "Daily_2051_2075"
        )
        write_dataframe(
            writer,
            daily_period_tables["2076-2100"],
            "Daily_2076_2100"
        )
        write_dataframe(
            writer,
            create_full_period_table(daily_results),
            "Daily_2026_2100"
        )

        write_dataframe(
            writer,
            annual_period_tables["2026-2050"],
            "Annual_2026_2050"
        )
        write_dataframe(
            writer,
            annual_period_tables["2051-2075"],
            "Annual_2051_2075"
        )
        write_dataframe(
            writer,
            annual_period_tables["2076-2100"],
            "Annual_2076_2100"
        )
        write_dataframe(
            writer,
            create_full_period_table(annual_results),
            "Annual_2026_2100"
        )

        write_dataframe(
            writer,
            daily_results,
            "Daily_Detailed"
        )
        write_dataframe(
            writer,
            annual_results,
            "Annual_Detailed"
        )
        write_dataframe(
            writer,
            national_summary,
            "National_Summary"
        )

    format_excel(output_file)

    print(f"\nResults exported successfully:\n{output_file}")

def main():

    print("=" * 70)
    print("Extreme Weather Impacts on Photovoltaic Potential")
    print("=" * 70)

    print(
        "\nNational summaries use cosine-latitude area weighting."
    )

    for scenario in SCENARIOS:

        print("\n" + "=" * 70)
        print(f"Processing climate scenario: {scenario}")
        print("=" * 70)

        try:
            daily_results, annual_results = analyze_scenario(scenario)

            export_scenario_results(
                scenario,
                daily_results,
                annual_results
            )

            print(f"\nCompleted: {scenario}")

        except Exception as error:
            print(f"\nError processing {scenario}: {error}")
            raise

    print("\nAll analyses completed.")

if __name__ == "__main__":
    main()
