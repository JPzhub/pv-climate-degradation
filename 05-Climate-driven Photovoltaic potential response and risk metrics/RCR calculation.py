# -*- coding: utf-8 -*-
#RCR

# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

# =========================================================
# 1. Configuration
# =========================================================

base_dir = r"result-PVpot tend\provinceandarea"

output_dir = r'M:\气候变化对光伏可变性影响\word\2026-NC投稿\data source\06_Figure_Generation'

scenarios = ["SSP126", "SSP245", "SSP585"]

scenario_label = {
    "SSP126": "SSP1-2.6",
    "SSP245": "SSP2-4.5",
    "SSP585": "SSP5-8.5"
}

# =========================================================
# 2. Province Mapping
# =========================================================

province_py = {
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

# =========================================================
# 3. Read and Process Scenario Data
# =========================================================

all_data = []

for scenario in scenarios:

    print(f"Processing {scenario}...")

    excel_path = os.path.join(
        base_dir,
        f"{scenario}_综合PVpot分析.xlsx"
    )

    if not os.path.exists(excel_path):
        raise FileNotFoundError(
            f"Input file not found: {excel_path}"
        )

    # Read the fourth worksheet
    df_rate = pd.read_excel(
        excel_path,
        sheet_name=3
    )

    # Validate input structure
    if df_rate.shape[1] != 4:
        raise ValueError(
            f"Unexpected number of columns in {excel_path}. "
            f"Expected 4 columns, found {df_rate.shape[1]}."
        )

    # Standardize column names
    df_rate.columns = [
        "Province",
        "Historical_Mean",
        "Future_Mean",
        "Relative_Change_Original"
    ]

    # Convert numerical columns
    numeric_cols = [
        "Historical_Mean",
        "Future_Mean",
        "Relative_Change_Original"
    ]

    for col in numeric_cols:
        df_rate[col] = pd.to_numeric(
            df_rate[col],
            errors="coerce"
        )

    # Calculate relative change rate
    df_rate["Relative_Change"] = np.where(
        df_rate["Historical_Mean"].notna()
        & (df_rate["Historical_Mean"] != 0)
        & df_rate["Future_Mean"].notna(),
        (
            df_rate["Future_Mean"]
            - df_rate["Historical_Mean"]
        ) / df_rate["Historical_Mean"],
        np.nan
    )

    # Add scenario information
    df_rate["Scenario"] = scenario
    df_rate["Scenario_Label"] = scenario_label[scenario]

    all_data.append(df_rate)

# =========================================================
# 4. Merge All Scenarios
# =========================================================

df_all = pd.concat(
    all_data,
    ignore_index=True
)

# Add province abbreviations
df_all["Province_Code"] = df_all["Province"].map(
    province_py
)

# Check unmatched provinces
unmatched = df_all.loc[
    df_all["Province_Code"].isna(),
    "Province"
].dropna().unique()

if len(unmatched) > 0:
    print("Warning: Unmatched provinces:")
    print(unmatched)

# =========================================================
# 5. Organize Output Columns
# =========================================================

output_columns = [
    "Province",
    "Province_Code",
    "Scenario",
    "Scenario_Label",
    "Historical_Mean",
    "Future_Mean",
    "Relative_Change",
    "Relative_Change_Original"
]

df_all = df_all[output_columns]

# Sort data
df_all = df_all.sort_values(
    by=["Scenario", "Province"]
).reset_index(drop=True)

# =========================================================
# 6. Export Results
# =========================================================

output_excel = os.path.join(
    output_dir,
    "Province_PVpot_Climate_Response_All_Scenarios.xlsx"
)

output_csv = os.path.join(
    output_dir,
    "Province_PVpot_Climate_Response_All_Scenarios.csv"
)

df_all.to_excel(
    output_excel,
    index=False,
    sheet_name="Province_Scenario_Data"
)

df_all.to_csv(
    output_csv,
    index=False,
    encoding="utf-8-sig"
)



