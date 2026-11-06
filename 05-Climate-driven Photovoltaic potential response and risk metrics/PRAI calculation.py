# -*- coding: utf-8 -*-
# PRAI

# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

# =========================================================
# 1. Configuration
# =========================================================

base_dir = r"M:\气候变化对光伏可变性影响\result-PVpot CV\provincedata-0929"

output_dir = r'M:\气候变化对光伏可变性影响\word\2026-NC投稿\data source\06_Figure_Generation'

scenarios = ["SSP126", "SSP245", "SSP585"]

scenario_label = {
    "SSP126": "SSP1-2.6",
    "SSP245": "SSP2-4.5",
    "SSP585": "SSP5-8.5"
}

expected_provinces = 31

# =========================================================
# 2. Province Name Standardization
# =========================================================

alias_map = {
    "新疆维吾尔自治区": "新疆",
    "广西壮族自治区": "广西",
    "宁夏回族自治区": "宁夏",
    "西藏自治区": "西藏",
    "内蒙古自治区": "内蒙古"
}

province_en = {
    "北京": "Beijing",
    "天津": "Tianjin",
    "河北": "Hebei",
    "山西": "Shanxi",
    "内蒙古": "Inner Mongolia",
    "辽宁": "Liaoning",
    "吉林": "Jilin",
    "黑龙江": "Heilongjiang",
    "上海": "Shanghai",
    "江苏": "Jiangsu",
    "浙江": "Zhejiang",
    "安徽": "Anhui",
    "福建": "Fujian",
    "江西": "Jiangxi",
    "山东": "Shandong",
    "河南": "Henan",
    "湖北": "Hubei",
    "湖南": "Hunan",
    "广东": "Guangdong",
    "广西": "Guangxi",
    "海南": "Hainan",
    "重庆": "Chongqing",
    "四川": "Sichuan",
    "贵州": "Guizhou",
    "云南": "Yunnan",
    "西藏": "Tibet",
    "陕西": "Shaanxi",
    "甘肃": "Gansu",
    "青海": "Qinghai",
    "宁夏": "Ningxia",
    "新疆": "Xinjiang"
}

province_code = {
    "北京": "BJ",
    "天津": "TJ",
    "河北": "HE",
    "山西": "SX",
    "内蒙古": "NM",
    "辽宁": "LN",
    "吉林": "JL",
    "黑龙江": "HL",
    "上海": "SH",
    "江苏": "JS",
    "浙江": "ZJ",
    "安徽": "AH",
    "福建": "FJ",
    "江西": "JX",
    "山东": "SD",
    "河南": "HA",
    "湖北": "HB",
    "湖南": "HN",
    "广东": "GD",
    "广西": "GX",
    "海南": "HI",
    "重庆": "CQ",
    "四川": "SC",
    "贵州": "GZ",
    "云南": "YN",
    "西藏": "XZ",
    "陕西": "SN",
    "甘肃": "GS",
    "青海": "QH",
    "宁夏": "NX",
    "新疆": "XJ"
}

region_map = {
    "北京": "NC",
    "天津": "NC",
    "河北": "NC",
    "山西": "NC",
    "内蒙古": "NC",
    "辽宁": "NE",
    "吉林": "NE",
    "黑龙江": "NE",
    "上海": "EC",
    "江苏": "EC",
    "浙江": "EC",
    "安徽": "EC",
    "福建": "EC",
    "江西": "EC",
    "山东": "EC",
    "河南": "CS",
    "湖北": "CS",
    "湖南": "CS",
    "广东": "CS",
    "广西": "CS",
    "海南": "CS",
    "重庆": "SW",
    "四川": "SW",
    "贵州": "SW",
    "云南": "SW",
    "西藏": "SW",
    "陕西": "NW",
    "甘肃": "NW",
    "青海": "NW",
    "宁夏": "NW",
    "新疆": "NW"
}

# =========================================================
# 3. Input Column Mapping
# =========================================================

column_mapping = {
    "极端高PVpot天数/年": "High_PV_Days",
    "极端低PVpot天数/年": "Low_PV_Days",
    "极端高PVpot趋势(天/十年)": "High_PV_Trend",
    "极端低PVpot趋势(天/十年)": "Low_PV_Trend"
}

required_input_columns = [
    "省份",
    *column_mapping.keys()
]

# =========================================================
# 4. Z-score Standardization
# =========================================================

def calculate_zscore(series):
    """
    Calculate Z-scores using the sample standard deviation.
    This preserves the original pandas Series.std() behavior.
    """

    mean_value = series.mean()
    std_value = series.std(ddof=1)

    if pd.isna(std_value) or std_value == 0:
        raise ValueError(
            f"Cannot standardize a constant or invalid variable: "
            f"{series.name}"
        )

    return (series - mean_value) / std_value


# =========================================================
# 5. Load and Validate Scenario Data
# =========================================================

def load_scenario_data(scenario):

    input_file = os.path.join(
        base_dir,
        f"{scenario}_省级综合分析结果.xlsx"
    )

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )

    df = pd.read_excel(input_file)

    missing_columns = [
        col for col in required_input_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {input_file}: {missing_columns}"
        )

    df = df[required_input_columns].copy()

    # Standardize province names
    df["省份"] = (
        df["省份"]
        .astype(str)
        .str.strip()
        .replace(alias_map)
    )

    # Rename input variables
    df = df.rename(columns=column_mapping)

    # Convert input variables to numeric values
    input_variables = list(column_mapping.values())

    for col in input_variables:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Check province uniqueness
    if df["省份"].duplicated().any():
        duplicated = df.loc[
            df["省份"].duplicated(keep=False),
            "省份"
        ].tolist()

        raise ValueError(
            f"Duplicate province records in {scenario}: {duplicated}"
        )

    # Check missing values
    if df[input_variables].isna().any().any():
        missing_summary = df[input_variables].isna().sum()

        raise ValueError(
            f"Missing numerical values in {scenario}:\n"
            f"{missing_summary}"
        )

    # Check province coverage
    if df["省份"].nunique() != expected_provinces:
        raise ValueError(
            f"{scenario} contains {df['省份'].nunique()} provinces; "
            f"expected {expected_provinces}."
        )

    # Add province information
    df["Province"] = df["省份"]
    df["Province_EN"] = df["Province"].map(province_en)
    df["Province_Code"] = df["Province"].map(province_code)
    df["Region_Code"] = df["Province"].map(region_map)

    mapping_columns = [
        "Province_EN",
        "Province_Code",
        "Region_Code"
    ]

    if df[mapping_columns].isna().any().any():
        raise ValueError(
            f"Unmatched province mapping detected in {scenario}."
        )

    df["Scenario"] = scenario
    df["Scenario_Label"] = scenario_label[scenario]

    return df


# =========================================================
# 6. Calculate PRAI
# =========================================================

def calculate_prai(df):

    df = df.copy()

    # Standardize high PV potential indicators
    df["Z_High_PV_Days"] = calculate_zscore(
        df["High_PV_Days"]
    )

    df["Z_High_PV_Trend"] = calculate_zscore(
        df["High_PV_Trend"]
    )

    # Standardize low PV potential indicators
    df["Z_Low_PV_Days"] = calculate_zscore(
        df["Low_PV_Days"]
    )

    df["Z_Low_PV_Trend"] = calculate_zscore(
        df["Low_PV_Trend"]
    )

    # Calculate hazard component
    df["Hazard"] = (
        df["Z_High_PV_Days"]
        + df["Z_High_PV_Trend"]
    )

    # Calculate constraint component
    df["Constraint"] = (
        df["Z_Low_PV_Days"]
        + df["Z_Low_PV_Trend"]
    )

    # Calculate Provincial Risk Amplification Index
    df["PRAI"] = (
        df["Hazard"]
        - df["Constraint"]
    )

    return df


# =========================================================
# 7. Process All Scenarios
# =========================================================

scenario_results = []

for scenario in scenarios:

    print(f"Processing {scenario}...")

    df_scenario = load_scenario_data(scenario)

    df_scenario = calculate_prai(df_scenario)

    scenario_results.append(df_scenario)

# Combine all scenarios
df_all = pd.concat(
    scenario_results,
    ignore_index=True
)

# =========================================================
# 8. Organize Output Data
# =========================================================

output_columns = [
    "Province",
    "Province_EN",
    "Province_Code",
    "Region_Code",
    "Scenario",
    "Scenario_Label",
    "High_PV_Days",
    "Low_PV_Days",
    "High_PV_Trend",
    "Low_PV_Trend",
    "Z_High_PV_Days",
    "Z_High_PV_Trend",
    "Z_Low_PV_Days",
    "Z_Low_PV_Trend",
    "Hazard",
    "Constraint",
    "PRAI"
]

df_all = df_all[output_columns].copy()

df_all = df_all.sort_values(
    by=["Scenario", "Province_Code"]
).reset_index(drop=True)

# =========================================================
# 9. Export Results
# =========================================================

output_csv = os.path.join(
    output_dir,
    "PRAI_Province_Scenario_Results.csv"
)

output_excel = os.path.join(
    output_dir,
    "PRAI_Province_Scenario_Results.xlsx"
)

df_all.to_csv(
    output_csv,
    index=False,
    encoding="utf-8-sig"
)

with pd.ExcelWriter(
    output_excel,
    engine="openpyxl"
) as writer:

    df_all.to_excel(
        writer,
        sheet_name="PRAI_All_Scenarios",
        index=False
    )

    for scenario in scenarios:

        df_scenario = df_all[
            df_all["Scenario"] == scenario
        ].copy()

        df_scenario.to_excel(
            writer,
            sheet_name=scenario,
            index=False
        )

# =========================================================
# 10. Quality Control
# =========================================================

print("\nPRAI calculation completed.")

print(f"Total records: {len(df_all)}")
print(f"Number of provinces: {df_all['Province'].nunique()}")
print(f"Number of scenarios: {df_all['Scenario'].nunique()}")

print("\nPRAI summary by scenario:")

print(
    df_all.groupby("Scenario")["PRAI"].agg(
        ["count", "mean", "std", "min", "max"]
    )
)

print(f"\nCSV output: {output_csv}")
print(f"Excel output: {output_excel}")
