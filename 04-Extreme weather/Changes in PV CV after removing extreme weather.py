
# -*- coding: utf-8 -*-

import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

data_dir_pvpot = (
    r"result-PVpot"
)

data_dir_climate = (
    r"K:\NEX-GDDP-CMIP6\sfcWind-csv-daily"
)

output_dir = (
    r"Changes in PV CV after removing extreme weather"
)

os.makedirs(output_dir, exist_ok=True)


SCENARIOS = ["SSP126", "SSP245", "SSP585"]

VARIABLES = {
    "Radiation": "rsds",
    "Temperature": "tas",
    "PVpot": "PVpot"
}


hist_start = 2015
hist_end = 2025

base_period = slice(
    f"{hist_start}-01-01",
    f"{hist_end}-12-31"
)

future_periods = {
    "2026-2050": (2026, 2050),
    "2051-2075": (2051, 2075),
    "2076-2100": (2076, 2100),
    "2026-2100": (2026, 2100)
}


# q = 0.1 corresponds to P10 and P90.
threshold_quantile = 0.1
MIN_VALID_DAYS = 2
MIN_HISTORICAL_DAYS = 365
MEAN_TOLERANCE = 1e-10

SHEET_PERIODS = "Grid_Analysis_By_Period"
SHEET_FULL = "Grid_Analysis_Full_Period"
SHEET_SUMMARY = "Scenario_Summary"

def load_processed_data(scenario, var_name):

    if var_name == "PVpot":
        filename = f"PVpot_{scenario}-2015-2100.csv"
        file_path = os.path.join(
            data_dir_pvpot,
            filename
        )

    else:
        filename = (
            f"{scenario}-2015-2100-{var_name}-日.csv"
        )

        file_path = os.path.join(
            data_dir_climate,
            filename
        )

    try:

        if not os.path.exists(file_path):

            raise FileNotFoundError(
                f"Input file not found: {file_path}"
            )

        print(f"\nLoading file: {file_path}")

        df = pd.read_csv(
            file_path,
            encoding="gbk"
        )

        print(f"Original columns: {df.columns.tolist()}")

        if "Unnamed: 0" in df.columns:

            df = df.drop(
                columns=["Unnamed: 0"]
            )

        required_cols = [
            "lat",
            "lon"
        ]

        missing_cols = [
            col for col in required_cols
            if col not in df.columns
        ]

        if missing_cols:

            raise ValueError(
                f"Missing required columns: {missing_cols}"
            )
        if "geometry" not in df.columns:

            warnings.warn(
                f"Geometry column missing in {filename}. "
                "Generating POINT geometries."
            )

            df["geometry"] = df.apply(
                lambda row: (
                    f"POINT ({row['lon']} {row['lat']})"
                ),
                axis=1
            )
        index_cols = [
            "lat",
            "lon",
            "geometry"
        ]

        date_cols = [
            col for col in df.columns
            if col not in index_cols
        ]

        if not date_cols:

            raise ValueError(
                "No valid daily date columns were found."
            )

        data = (
            df.set_index(index_cols)[date_cols]
            .T
        )


        parsed_dates = pd.to_datetime(
            data.index,
            format="mixed",
            errors="coerce"
        )

        if parsed_dates.isna().any():

            invalid_dates = data.index[
                parsed_dates.isna()
            ]

            raise ValueError(
                "Invalid date labels detected: "
                f"{list(invalid_dates[:5])}"
            )

        data.index = parsed_dates
        data.index.name = "Date"
        data = data.apply(
            pd.to_numeric,
            errors="coerce"
        )

        data = data.sort_index()

        if data.index.has_duplicates:

            duplicated_dates = (
                data.index[
                    data.index.duplicated()
                ]
                .unique()
            )

            raise ValueError(
                "Duplicated dates detected: "
                f"{list(duplicated_dates[:5])}"
            )

        if var_name == "tas":

            data = data - 273.15
        hist_data = data.loc[base_period]

        return data

    except Exception as error:

        print(f"Failed to load file: {filename}" )

        print(f"Error details: {str(error)}")
        raise
def align_input_data(pvpot_data, radiation_data, temperature_data):
    common_columns = (
        pvpot_data.columns
        .intersection(radiation_data.columns)
        .intersection(temperature_data.columns)
    )

    if len(common_columns) == 0:

        raise ValueError("No common grid cells were found." )
    common_columns = [
        col for col in pvpot_data.columns
        if col in set(common_columns)
    ]

    pvpot_data = (
        pvpot_data
        .loc[:, common_columns]
        .sort_index()
    )

    radiation_data = radiation_data.reindex(
        index=pvpot_data.index,
        columns=common_columns
    )

    temperature_data = temperature_data.reindex(
        index=pvpot_data.index,
        columns=common_columns
    )
   
    return(pvpot_data, radiation_data,  temperature_data, common_columns)


def calculate_historical_thresholds(
    radiation_series,
    temperature_series
):
    """
    Calculate grid-specific historical P10 and P90 thresholds.
    Historical reference period:        2015-2025

    Thresholds are calculated independently for each grid cell.
    """

    hist_rad = (
        radiation_series
        .loc[base_period]
        .dropna()
    )

    hist_temp = (
        temperature_series
        .loc[base_period]
        .dropna()
    )

    if (
        len(hist_rad) < MIN_HISTORICAL_DAYS
        or len(hist_temp) < MIN_HISTORICAL_DAYS
    ):

        return None

    thresholds = {

        "rad_p90": hist_rad.quantile(
            1 - threshold_quantile
        ),

        "rad_p10": hist_rad.quantile(
            threshold_quantile
        ),

        "temp_p90": hist_temp.quantile(
            1 - threshold_quantile
        ),

        "temp_p10": hist_temp.quantile(
            threshold_quantile
        )

    }

    if not all(
        np.isfinite(value)
        for value in thresholds.values()
    ):

        return None

    return thresholds

def calculate_cv(series):
    """
    Calculate the coefficient of variation.

    CV = Sample standard deviation / Arithmetic mean
    """

    valid_series = (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .dropna()
    )

    if len(valid_series) < MIN_VALID_DAYS:

        return np.nan

    mean_value = valid_series.mean()

    std_value = valid_series.std(
        ddof=1
    )

    if (
        not np.isfinite(mean_value)
        or not np.isfinite(std_value)
        or abs(mean_value) < MEAN_TOLERANCE
    ):

        return np.nan

    return float(
        std_value / mean_value
    )


def calculate_cv_by_year(series):
    if series.empty:
        return pd.Series(
            dtype=float
        )

    valid_series = series.dropna()

    if valid_series.empty:
        return pd.Series(
            dtype=float
        )

    cv_results = {}
    for year, annual_data in valid_series.groupby(
        valid_series.index.year ):

        cv_results[int(year)] = calculate_cv(
            annual_data)

    return pd.Series(
        cv_results,
        dtype=float
    ).sort_index()


def build_scenario_masks(
    radiation_series,
    temperature_series,
    valid_climate_mask,
    thresholds
):
    rad_high = (
        radiation_series
        > thresholds["rad_p90"]
    )

    rad_low = (
        radiation_series
        < thresholds["rad_p10"]
    )

    temp_high = (
        temperature_series
        > thresholds["temp_p90"]
    )

    temp_low = (
        temperature_series
        < thresholds["temp_p10"]
    )

    scenario_masks = {

        "High_Radiation": (
            valid_climate_mask
            & ~rad_high
        ),

        "Low_Radiation": (
            valid_climate_mask
            & ~rad_low
        ),

        "High_Temperature": (
            valid_climate_mask
            & ~temp_high
        ),

        "Low_Temperature": (
            valid_climate_mask
            & ~temp_low
        ),

        "Radiation_Extremes": (
            valid_climate_mask
            & ~(rad_high | rad_low)
        ),

        "Temperature_Extremes": (
            valid_climate_mask
            & ~(temp_high | temp_low)
        ),

        "All_Extremes": (
            valid_climate_mask
            & ~(
                rad_high
                | rad_low
                | temp_high
                | temp_low
            )
        )

    }

    return scenario_masks

class CVAnalyzer:

    def __init__(self):
        pass

    def calculate_scenarios(
        self,
        grid_point,
        yield_s_full,
        rad_s_full,
        temp_s_full,
        period_name,
        start,
        end
    ):
        thresholds = calculate_historical_thresholds(
            rad_s_full,
            temp_s_full
        )

        if thresholds is None:

            return []

        period_slice = slice(
            f"{start}-01-01",
            f"{end}-12-31"
        )

        period_yield = (
            yield_s_full
            .loc[period_slice]
        )

        if period_yield.empty:

            return []

        period_rad = (
            rad_s_full
            .reindex(period_yield.index)
        )

        period_temp = (
            temp_s_full
            .reindex(period_yield.index)
        )

        valid_climate_mask = (
            period_yield.notna()
            & period_rad.notna()
            & period_temp.notna()
        )

        if valid_climate_mask.sum() == 0:

            return []
        baseline_yield = period_yield.where(
            valid_climate_mask
        )

        original_cv_by_year = (
            calculate_cv_by_year(
                baseline_yield
            )
        )
        scenario_masks = build_scenario_masks(
            radiation_series=period_rad,
            temperature_series=period_temp,
            valid_climate_mask=valid_climate_mask,
            thresholds=thresholds
        )

        lat = grid_point[0]
        lon = grid_point[1]
        geometry = grid_point[2]

        results = []

        for scenario_name, retention_mask in (
            scenario_masks.items()
        ):

            filtered_yield = baseline_yield.where(
                retention_mask
            )

            new_cv_by_year = (
                calculate_cv_by_year(
                    filtered_yield
                )
            )

            retained_count = int(
                filtered_yield.notna().sum()
            )

            original_count = int(
                baseline_yield.notna().sum()
            )

            print(
                f"{scenario_name}: "
                f"Original={original_count}, "
                f"Retained={retained_count}"
            )

            for year in original_cv_by_year.index:

                original_cv = (
                    original_cv_by_year.get(
                        year,
                        np.nan
                    )
                )

                new_cv = (
                    new_cv_by_year.get(
                        year,
                        np.nan
                    )
                )

                if (
                    pd.isna(original_cv)
                    or pd.isna(new_cv)
                    or not np.isfinite(original_cv)
                    or not np.isfinite(new_cv)
                    or abs(original_cv) < MEAN_TOLERANCE
                ):

                    change_pct = np.nan

                else:

                    change_pct = (
                        (new_cv - original_cv)
                        / original_cv
                    ) * 100.0

                results.append({

                    "lat": lat,

                    "lon": lon,

                    "geometry": geometry,

                    "Period": period_name,

                    "Scenario": scenario_name,

                    "Year": int(year),

                    "Original_CV": (
                        round(float(original_cv), 4)
                        if pd.notna(original_cv)
                        else np.nan
                    ),

                    "Filtered_CV": (
                        round(float(new_cv), 4)
                        if pd.notna(new_cv)
                        else np.nan
                    ),

                    "CV_Change_Percent": (
                        round(float(change_pct), 2)
                        if pd.notna(change_pct)
                        else np.nan
                    )

                })

        return results

def create_wide_format(
    long_df,
    include_period=True
):

    if long_df.empty:

        print(
            "Warning: empty long-format DataFrame."
        )

        return pd.DataFrame()

    index_cols = (

        ["lat", "lon", "geometry", "Period"]

        if include_period

        else ["lat", "lon", "geometry"]

    )

    scenarios = (
        long_df["Scenario"]
        .dropna()
        .unique()
        .tolist()
    )

    metrics = [
        "Original_CV",
        "Filtered_CV",
        "CV_Change_Percent"
    ]

    grouped = (
        long_df
        .groupby(
            index_cols + ["Scenario"],
            dropna=False
        )
        .agg({

            "Original_CV": "mean",

            "Filtered_CV": "mean"

        })
        .reset_index()
    )

    denominator = (
        grouped["Original_CV"]
        .replace(0, np.nan)
    )

    grouped["CV_Change_Percent"] = (

        (
            grouped["Filtered_CV"]
            - grouped["Original_CV"]
        )
        / denominator

    ) * 100.0


    pivot_df = grouped.pivot(
        index=index_cols,
        columns="Scenario",
        values=metrics
    )

    wide_df = pd.DataFrame(
        index=pivot_df.index
    )

    for scenario_name in scenarios:

        for metric in metrics:

            key = (
                metric,
                scenario_name
            )

            column_name = (
                f"{scenario_name}_{metric}"
            )

            if key in pivot_df.columns:

                wide_df[column_name] = (
                    pivot_df[key]
                )

            else:

                wide_df[column_name] = np.nan

    wide_df = wide_df.reset_index()

    return wide_df

def calculate_scenario_summary(long_df):
    """
    Summarize scenario-level results.

    Calculate the mean annual CV for each grid cell and period.
    Calculate spatial mean and standard deviation across grid cells.
    """

    if long_df.empty:

        print(
            "Warning: empty DataFrame for scenario summary."
        )

        return pd.DataFrame()
    grid_period = (

        long_df

        .groupby(
            [
                "lat",
                "lon",
                "geometry",
                "Period",
                "Scenario"
            ],
            dropna=False
        )

        .agg({

            "Original_CV": "mean",

            "Filtered_CV": "mean"

        })

        .reset_index()

    )

    denominator = (
        grid_period["Original_CV"]
        .replace(0, np.nan)
    )

    grid_period["CV_Change_Percent"] = (

        (
            grid_period["Filtered_CV"]
            - grid_period["Original_CV"]
        )
        / denominator

    ) * 100.0

    summary = (

        grid_period

        .groupby(
            ["Period", "Scenario"]
        )

        .agg({

            "Original_CV": ["mean", "std"],

            "Filtered_CV": ["mean", "std"],

            "CV_Change_Percent": ["mean", "std"]

        })

        .reset_index()

    )

    summary.columns = [

        "Period",

        "Scenario",

        "Original_CV_Mean",

        "Original_CV_SD",

        "Filtered_CV_Mean",

        "Filtered_CV_SD",

        "CV_Change_Mean",

        "CV_Change_SD"

    ]

    numeric_cols = [

        "Original_CV_Mean",

        "Original_CV_SD",

        "Filtered_CV_Mean",

        "Filtered_CV_SD",

        "CV_Change_Mean",

        "CV_Change_SD"

    ]

    summary[numeric_cols] = (
        summary[numeric_cols].round(4)
    )

    return summary

def format_worksheet(
    worksheet,
    workbook,
    dataframe,
    include_period=True
):

    header_format = workbook.add_format({

        "bold": True,

        "border": 1,

        "bg_color": "#DDEBF7",

        "align": "center",

        "valign": "vcenter"

    })

    number_format = workbook.add_format({

        "num_format": "0.0000"

    })

    for col_idx, column_name in enumerate(
        dataframe.columns
    ):

        worksheet.write(
            0,
            col_idx,
            column_name,
            header_format
        )

    worksheet.set_column(
        0,
        2,
        18
    )

    if include_period:

        worksheet.set_column(
            3,
            3,
            15
        )

        start_col = 4

    else:

        start_col = 3

    if len(dataframe.columns) > start_col:

        worksheet.set_column(
            start_col,
            len(dataframe.columns) - 1,
            18,
            number_format
        )

    worksheet.freeze_panes(
        1,
        start_col
    )

    if len(dataframe.columns) > 0:

        worksheet.autofilter(
            0,
            0,
            max(len(dataframe), 1),
            len(dataframe.columns) - 1
        )


def save_scenario_sheet(
    writer,
    summary_df
):

    sheet_name = SHEET_SUMMARY

    summary_df.to_excel(
        writer,
        sheet_name=sheet_name,
        index=False
    )

    workbook = writer.book

    worksheet = writer.sheets[
        sheet_name
    ]

    header_format = workbook.add_format({

        "bold": True,

        "border": 1,

        "bg_color": "#DDEBF7",

        "align": "center"

    })

    number_format = workbook.add_format({

        "num_format": "0.0000"

    })

    for col_idx, column_name in enumerate(
        summary_df.columns
    ):

        worksheet.write(
            0,
            col_idx,
            column_name,
            header_format
        )

    worksheet.set_column(
        0,
        0,
        15
    )

    worksheet.set_column(
        1,
        1,
        24
    )

    worksheet.set_column(
        2,
        7,
        18,
        number_format
    )

    worksheet.freeze_panes(
        1,
        2
    )


def export_results(
    scenario,
    wide_df_periods,
    wide_df_full,
    summary_df
):

    output_path = os.path.join(

        output_dir,

        f"{scenario}_analysis.xlsx"

    )

    with pd.ExcelWriter(

        output_path,

        engine="xlsxwriter"

    ) as writer:

        wide_df_periods.to_excel(

            writer,

            sheet_name=SHEET_PERIODS,

            index=False

        )

        wide_df_full.to_excel(

            writer,

            sheet_name=SHEET_FULL,

            index=False

        )

        save_scenario_sheet(
            writer,
            summary_df
        )

        workbook = writer.book

        worksheet_periods = writer.sheets[
            SHEET_PERIODS
        ]

        worksheet_full = writer.sheets[
            SHEET_FULL
        ]

        format_worksheet(

            worksheet_periods,

            workbook,

            wide_df_periods,

            include_period=True

        )

        format_worksheet(

            worksheet_full,

            workbook,

            wide_df_full,

            include_period=False

        )

    if not os.path.exists(output_path):

        raise IOError(
            f"Output file was not created: {output_path}"
        )

    return output_path


def analyze_scenario(scenario):

    print("\n" + "=" * 75)

    print(
        f"Processing scenario: {scenario}"
    )

    print("=" * 75)
    print("\nLoading input datasets...")

    yield_data = load_processed_data(
        scenario,
        "PVpot"
    )

    rad_data = load_processed_data(
        scenario,
        "rsds"
    )

    temp_data = load_processed_data(
        scenario,
        "tas"
    )

    (
        yield_data,
        rad_data,
        temp_data,
        common_columns
    ) = align_input_data(

        yield_data,

        rad_data,

        temp_data

    )

    print(
        f"\nNumber of common grid cells: "
        f"{len(common_columns)}"
    )

    analyzer = CVAnalyzer()

    results_periods = []

    results_full = []

    print("\nStarting grid-level analysis...")

    for grid_point in tqdm(
        common_columns,
        desc=f"{scenario} grid analysis"
    ):

        yield_series = yield_data[
            grid_point
        ]

        rad_series = rad_data[
            grid_point
        ]

        temp_series = temp_data[
            grid_point
        ]

        for period_name, (
            start,
            end
        ) in future_periods.items():

            results = analyzer.calculate_scenarios(

                grid_point=grid_point,

                yield_s_full=yield_series,

                rad_s_full=rad_series,

                temp_s_full=temp_series,

                period_name=period_name,

                start=start,

                end=end

            )

            results_periods.extend(
                results
            )

            if period_name == "2026-2100":

                results_full.extend(
                    results
                )

    output_columns = [

        "lat",

        "lon",

        "geometry",

        "Period",

        "Scenario",

        "Year",

        "Original_CV",

        "Filtered_CV",

        "CV_Change_Percent"

    ]

    long_df_periods = pd.DataFrame(

        results_periods,

        columns=output_columns

    )

    long_df_full = pd.DataFrame(

        results_full,

        columns=output_columns

    )

    if long_df_periods.empty:

        raise RuntimeError(

            f"No valid annual CV records were generated "
            f"for {scenario}. Check input data and date coverage."

        )

    print("\nConverting results to wide format...")

    wide_df_periods = create_wide_format(

        long_df_periods,

        include_period=True

    )

    wide_df_full = create_wide_format(

        long_df_full,

        include_period=False

    )

    summary_df = calculate_scenario_summary(
        long_df_periods
    )

    print("\nExporting results...")

    output_path = export_results(

        scenario,

        wide_df_periods,

        wide_df_full,

        summary_df

    )

    return output_path

def main():
    print("=" * 75)
    print(f"Scenarios: {SCENARIOS}") 

    completed_scenarios = []

    for scenario in SCENARIOS:

        try:

            output_path = analyze_scenario(
                scenario
            )

            completed_scenarios.append(
                (scenario, output_path)
            )

        except Exception as error:

            print(
                f"\nScenario processing failed: {scenario}"
            )

            print(
                f"Error: {str(error)}"
            )

            raise

    print("\n" + "=" * 75)

    print("All analyses completed.")

    print("=" * 75)

    for scenario, output_path in completed_scenarios:

        print(
            f"{scenario}: {output_path}"
        )


if __name__ == "__main__":

    main()
