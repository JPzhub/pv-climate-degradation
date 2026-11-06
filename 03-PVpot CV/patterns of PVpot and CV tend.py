# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np

input_dir = r"result-PVpot CV"
output_dir = r"result-patterns of PVpot and CV tend"
os.makedirs(output_dir, exist_ok=True)

SCENARIOS = ['SSP126', 'SSP245', 'SSP585']
TREND_TYPES = [
    'PVpotUP_CVUP', 'PVpotUP_CVDOWN', 'PVpotDOWN_CVUP', 'PVpotDOWN_CVDOWN', 'other', 'error'
]

def load_grid_data(scenario):
    filename = f"{scenario}_grid_analysis.csv"
    input_path = os.path.join(input_dir, filename)    
    try:
        df = pd.read_csv(input_path, encoding='gbk')
        print(f"{input_path}")
        return df
    except Exception as e:
        print(f"{input_path}")
        print(f"{str(e)}")
        return None
def classify_trend(row):
    try:
        if pd.notnull(row['PVpot_MK-p']) and row['PVpot_MK-p'] < 0.05:
            yield_trend = 'UP' if row['PVpot_Sen(%/decade)'] > 0 else 'DOWN'
        else:
            yield_trend = 'noSig'
        if pd.notnull(row['CVp']) and row['CVp'] < 0.05:
            cv_trend = 'UP' if row['CVslop(/decade)'] > 0 else 'DOWN'
        else:
            cv_trend = 'noSig'

        if yield_trend == 'UP' and cv_trend == 'UP':
            return 'PVpotUP_CVUP'
        elif yield_trend == 'UP' and cv_trend == 'DOWN':
            return 'PVpotUP_CVDOWN'
        elif yield_trend == 'DOWN' and cv_trend == 'UP':
            return 'PVpotUP_CVDOWN'
        elif yield_trend == 'DOWN' and cv_trend == 'DOWN':
            return 'PVpotDOWN_CVDOWN'
        else:
            return 'other'
    except:
        return 'error'

def analyze_scenario(scenario):

    df = load_grid_data(scenario)
    if df is None:
        return

    df['TrendCa'] = df.apply(classify_trend, axis=1)
    type_counts = df['TrendCa'].value_counts().reindex(TREND_TYPES, fill_value=0)
    output_path = os.path.join(output_dir, f"{scenario}TrendCategory.xlsx")
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        pd.DataFrame({
            'TrendCa': type_counts.index,
            'COUNT': type_counts.values
        }).to_excel(writer, sheet_name='stati', index=False)
        for trend_type in TREND_TYPES:
            type_df = df[df['TrendCa'] == trend_type]
            sheet_name = f"{scenario}_{trend_type}"[:31]
            if not type_df.empty:
                type_df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                pd.DataFrame(columns=df.columns).to_excel(writer, sheet_name=sheet_name, index=False)
        workbook = writer.book
        fmt = workbook.add_format({'text_wrap': True, 'valign': 'top'})
        num_fmt = workbook.add_format({'num_format': '0.0000'})
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column('A:C', 15, fmt)  # lat, lon, geometry
            worksheet.set_column('D:Z', 12, num_fmt)     
    print(f"{output_path}")
if __name__ == "__main__":
    try:
        import xlsxwriter
    except ImportError:
        sys.exit(1)    
    for scenario in SCENARIOS:
        print(f"{scenario}")
        analyze_scenario(scenario)    
    print("done")
