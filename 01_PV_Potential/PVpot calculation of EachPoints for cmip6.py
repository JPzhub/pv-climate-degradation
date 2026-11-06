# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 14:36:14 2025

@author: T&J
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Daily CSVs (2015-2100, per {scenario}) converted from original NEX-GDDP-CMIP6
# NetCDF files; a multi-model ensemble mean was applied over GFDL-CM4,
# GFDL-ESM4, IPSL-CM6A-LR, MPI-ESM1-2-HR/LR, MRI-ESM2-0, and NorESM2-LM.
# path
INPUT_DIR = r"K:\NEX-GDDP-CMIP6\sfcWind-csv-daily"
OUTPUT_DIR = r"M:\result-PVpot\point_result"
SCENARIOS = ['SSP126', 'SSP245', 'SSP585']
VARIABLES = {
    'wind': {'col': 'sfcWind', 'file': '{scenario}-2015-2100-sfcWind-daily.csv'},
    'rad': {'col': 'rsds', 'file': '{scenario}-2015-2100-rsds-daily.csv'},
    'temp': {'col': 'tas', 'file': '{scenario}-2015-2100-tas-daily.csv'}
}

def validate_scenario(scenario):
    valid_files = {}
    for var_type, var_info in VARIABLES.items():
        file_path = os.path.abspath(os.path.join(INPUT_DIR, var_info['file'].format(scenario=scenario)))
        
        print(f"{var_type}: {file_path}")
        if not os.path.exists(file_path):
            print(f"{scenario}  {var_type} : {file_path}")
            return None
        valid_files[var_type] = file_path
    return valid_files

def load_clean_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df = df[[col for col in df.columns if not col.startswith('Unnamed')]]
        required_cols = ['lat', 'lon']
        if not all(col in df.columns for col in required_cols):
            print(f"{os.path.basename(file_path)}: {required_cols}")
            return None, None
        if 'geometry' not in df.columns:
            df['geometry'] = df.apply(lambda row: f"POINT ({row['lon']} {row['lat']})", axis=1)
        date_cols = [col for col in df.columns if col not in ['lat', 'lon', 'geometry']]
        if not date_cols:
            return None, None

        df['lon_lat'] = df['lon'].astype(str) + '_' + df['lat'].astype(str) 
        metadata = df[['lon_lat', 'lat', 'lon', 'geometry']].drop_duplicates().set_index('lon_lat')
        data = df.set_index('lon_lat')[date_cols]
        return data, metadata
    except Exception as e:
        return None, None

def calculate_pvpot(scenario_files):
    wind_df, wind_meta = load_clean_data(scenario_files['wind'])
    rad_df, rad_meta = load_clean_data(scenario_files['rad'])
    temp_df, temp_meta = load_clean_data(scenario_files['temp'])
    if wind_df is None or rad_df is None or temp_df is None:
        return None, None
    common_columns = wind_df.columns.intersection(rad_df.columns).intersection(temp_df.columns)
    common_lon_lat = wind_df.index.intersection(rad_df.index).intersection(temp_df.index)
    
    if len(common_columns) == 0 or len(common_lon_lat) == 0:
        return None, None
    
    wind_df = wind_df.loc[common_lon_lat, common_columns]
    rad_df = rad_df.loc[common_lon_lat, common_columns]
    temp_df = temp_df.loc[common_lon_lat, common_columns]
    temp_df = temp_df - 273.15  # K to °C

    metadata = wind_meta.loc[common_lon_lat]
    if not (metadata.equals(rad_meta.loc[common_lon_lat]) and metadata.equals(temp_meta.loc[common_lon_lat])):
        return None, None

    result_df = pd.DataFrame(index=common_lon_lat, columns=common_columns)
    for date in tqdm(common_columns, desc="PVpot"):
        try:
            df = pd.DataFrame({
                'sfcWind': wind_df[date],
                'rsds': rad_df[date],
                'tas': temp_df[date]
            })

            df['T_cell'] = 4.3 + 0.943 * df['tas'] + 0.028 * df['rsds'] - 1.528 * df['sfcWind']
            
            result_df[date] = (1 - 0.005 * (df['T_cell'] - 25)) * (df['rsds'] / 1000)
            
        except Exception as e:
            print(f"{date} - {str(e)}")
    
    return result_df.round(4), metadata

def process_scenario(scenario):
    scenario_files = validate_scenario(scenario)
    if not scenario_files:
        return   
    result_df, metadata = calculate_pvpot(scenario_files)
    if result_df is None or result_df.empty:
        return
    result_wide = result_df.merge(metadata[['lat', 'lon', 'geometry']], left_index=True, right_index=True)
    result_wide = result_wide.reset_index(drop=True)[['lat', 'lon', 'geometry'] + list(result_df.columns)]

    output_path = os.path.abspath(os.path.join(OUTPUT_DIR, f"PVpot_{scenario}-2015-2100.csv"))
    try:
        result_wide.to_csv(output_path, index=False)
        print(f"{output_path}")
    except Exception as e:
        print(f"{str(e)}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for scenario in SCENARIOS:
        process_scenario(scenario)
    print("done")