
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from openpyxl.utils import get_column_letter

data_dir = r"NEX-GDDP-CMIP6\sfcWind-csv-daily"
output_dir = r"result-Extreme weather"
os.makedirs(output_dir, exist_ok=True)

SCENARIOS = ['SSP126', 'SSP245', 'SSP585']
VARIABLES = {'1': 'rsds', '2': 'tas'}
hist_start = 2015
hist_end = 2025

future_periods = {
    '2026-2050': (2026, 2050),
    '2051-2075': (2051, 2075),
    '2076-2100': (2076, 2100),
    '2026-2100': (2026, 2100)
}
extreme_types = {'high': 0.9, 'low': 0.1}


def load_analysis_data(scenario, variable):

    filename = f"{scenario}-2015-2100-{variable}-日.csv"
    file_path = os.path.join(data_dir, filename)
    
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path}")        
        df = pd.read_csv(file_path, encoding='gbk')
        print(f" {file_path}")
        print(f"{df.columns.tolist()}")
        print(f"{df.head(3)}\n")

        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])        
        required_cols = ['lat', 'lon']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"{required_cols},{df.columns.tolist()}")

        index_cols = required_cols.copy()
        if 'geometry' in df.columns:
            index_cols.append('geometry')
        date_cols = [col for col in df.columns if col not in index_cols]
        if not date_cols:
            raise ValueError("")        
        data = df.set_index(index_cols)[date_cols].T
        data.index = pd.to_datetime(data.index, format='mixed', errors='coerce')
        if data.index.isna().any():
            raise ValueError("error")        
        if variable == 'tas':
            data = data - 273.15
        return data    
    except Exception as e:
        print(f"{filename}")
        print(f"{str(e)}")
        return None
def full_analysis(scenario):
    final_results = []    
    for var_name, var_code in VARIABLES.items():
        print(f"{var_name}")        
        df = load_analysis_data(scenario, var_code)
        if df is None:
            continue            
        hist_data = df.loc[f"{hist_start}":f"{hist_end}"]
        future_data = df.loc[f"{future_periods['2026-2050'][0]}":]
        thresholds = {
            'high': hist_data.quantile(extreme_types['high']),
            'low': hist_data.quantile(extreme_types['low'])
        }

        for period_name, (start, end) in future_periods.items():
            print(f"{period_name}")
            period_data = future_data.loc[f"{start}":f"{end}"]
            high_events = (period_data > thresholds['high']).mean()
            low_events = (period_data < thresholds['low']).mean()
            for grid_point in period_data.columns:
                index_values = grid_point if isinstance(grid_point, tuple) else (grid_point,)
                result_dict = {
                    'var': var_name,
                    'lat': index_values[0],
                    'lon': index_values[1],
                    'time': period_name,
                    'extremeHigh(%)': round(high_events[grid_point], 2),
                    'extremeLow(%)': round(low_events[grid_point], 2)
                }
                if len(index_values) > 2: 
                    result_dict['geometry'] = index_values[2]
                final_results.append(result_dict)    
    return pd.DataFrame(final_results)
if __name__ == "__main__":
    try:
        import openpyxl
    except ImportError:
        import sys
        sys.exit(1)
    
    for scenario in SCENARIOS:
        print(f"{scenario}")
        result_df = full_analysis(scenario)
        if result_df.empty:
            continue

        summary = result_df.groupby(['var', 'time']).agg({
            'extremeHigh(%)': ['mean', 'std'],
            'extremeLow(%)': ['mean', 'std']
        })
        output_file = os.path.join(output_dir, f"{scenario}_extreme.xlsx")
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for var in VARIABLES.keys():
                var_df = result_df[result_df['var'] == var]
                index_cols = ['lat', 'lon']
                if 'geometry' in result_df.columns:
                    index_cols.append('geometry')
                reshaped = var_df.pivot_table(
                    index=index_cols,
                    columns='time',
                    values=['extremeHigh(%)', 'extremeLow(%)'],
                    aggfunc='first'
                )
                reshaped = reshaped.reset_index()
                reshaped.to_excel(writer, sheet_name=var, index=False)
            summary = result_df.groupby(['var', 'time'], as_index=False).agg(
                extremeHigh_mean=('extremeHigh(%)', lambda x: round(x.mean(), 2)),
                extremeHigh_std=('extremeHigh(%)', lambda x: round(x.std(), 2)),
                extremeLow_mean=('extremeLow(%)', lambda x: round(x.mean(), 2)),
                extremeLow_std=('extremeLow(%)', lambda x: round(x.std(), 2))
            )
            summary.rename(columns={
                'extremeHigh_mean': 'extremeHigh_mean (%)',
                'extremeHigh_std': 'extremeHigh_std (%)',
                'extremeLow_mean': 'extremeLow_mean (%)',
                'extremeLow_std': 'extremeLow_std (%)'
            }, inplace=True)
            summary['var'] = pd.Categorical(
                summary['var'],
                categories=['Daily Temperature', 'Daily Radiation'],
                ordered=True
            )
            summary.sort_values(['var', 'time']).to_excel(
                writer, 
                sheet_name='National Summary', 
                index=False,
                columns=[
                    'var', 'time',
                    'extremeHigh_mean (%)', 'extremeHigh_std (%)',
                    'extremeLow_mean (%)', 'extremeLow_std (%)'
                ]
            )
            
            # Set column widths via openpyxl column_dimensions.
            workbook = writer.book
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                # Number of columns written to the current sheet.
                df = reshaped if sheet_name in VARIABLES.keys() else summary
                num_cols = len(df.columns)
                for col_idx in range(1, num_cols + 1):
                    col_letter = get_column_letter(col_idx)
                    if col_idx <= (3 if 'geometry' in df.columns else 2):  # lat, lon, geometry
                        worksheet.column_dimensions[col_letter].width = 15
                    else:
                        worksheet.column_dimensions[col_letter].width = 12

        print(f"\nResults saved to: {output_file}")
        print("Sheets included:")
        print("- Daily Radiation\n- Daily Temperature\n- National Summary")