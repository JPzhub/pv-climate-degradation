# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 19:14:17 2025

@author: T&J
"""


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# ====================path ====================
# CSVs grouped by variable, converted from ERA5-Land monthly NetCDF files;
# each monthly .nc covers all days within that month (daily values).
INPUT_BASE_DIR = Path(r"csv_by_var")
OUTPUT_DIR = Path(r"point_result_ERA5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
YEARS = list(range(2015, 2026))
VARIABLES = ["u10", "v10", "t2m", "ssrd"]
VAR_FOLDERS = {var: INPUT_BASE_DIR / var for var in VARIABLES}
# ==============================================

def get_all_monthly_csv_files():
    all_months_data = {}
    for year in YEARS:
        for month in range(1, 13):
            date_str = f"{year}-{month:02d}"
            month_data = {}

            all_exist = True
            for var in VARIABLES:
                csv_folder = VAR_FOLDERS[var]
                csv_filename = f"ERA5_{var}_{date_str}.csv"
                csv_path = csv_folder / csv_filename

                if not csv_path.exists():
                    print(f"⚠️ missing: {csv_path}")
                    all_exist = False
                    break

                try:
                    df = pd.read_csv(csv_path)
                    month_data[var] = df
                except Exception as e:
                    print(f"{csv_path} failed: {e}")
                    all_exist = False
                    break

            if all_exist:
                all_months_data[date_str] = month_data
            else:
                print(f"⚠️{date_str}")
    return all_months_data

def compute_pv_for_group(group, date_columns):
    lat = group['lat'].iloc[0]
    lon = group['lon'].iloc[0]
    geometry = group['geometry'].iloc[0]

    meta_row = [round(float(lat), 4), round(float(lon), 4), f"POINT ({round(float(lon), 4)} {round(float(lat), 4)})"]
    pv_vals = []

    for date in date_columns:
        try:
            u10 = group.loc[group['lon'] == lon, date].iloc[0]
            v10 = group.loc[group['lon'] == lon, date].iloc[0]
            t2m = group.loc[group['lon'] == lon, date].iloc[0]
            ssrd = group.loc[group['lon'] == lon, date].iloc[0]
        except Exception:
            pv_vals.append(np.nan)
            continue

        try:
            u10 = float(u10)
            v10 = float(v10)
            t2m = float(t2m)
            ssrd = float(ssrd)
        except:
            pv_vals.append(np.nan)
            continue
        sfcWind = np.sqrt(u10**2 + v10**2)
        tas_c = t2m - 273.15
        rsds_wm2 = ssrd / 86400.0

        T_cell = 4.3 + 0.943 * tas_c + 0.028 * rsds_wm2 + 1.528 * sfcWind
        pv = (1 - 0.005 * (T_cell - 25)) * (rsds_wm2 / 1000)
        pv_vals.append(round(float(pv), 6))

    return meta_row + pv_vals

def main():
    print("1")
    all_months_data = get_all_monthly_csv_files()

    if not all_months_data:
        print("please check")
        return

    print(f"find {len(all_months_data)}，starting calculate PVpot...")

    all_results = []

    date_columns = sorted(all_months_data.keys())
    for date in tqdm(date_columns, desc="date"):
        month_data = all_months_data[date]
        if not month_data:
            continue
        dfs = []
        for var in VARIABLES:
            df = month_data[var]
            dfs.append(df)
        try:
            merged_df = dfs[0]
            for df in dfs[1:]:
                merged_df = pd.merge(merged_df, df, on=['lon', 'lat', 'geometry'], how='inner')
        except Exception as e:
            print(f"{date}）: {e}")
            continue
        if merged_df.empty:
            continue
        grouped = merged_df.groupby(['lat', 'lon', 'geometry'])

        for (lat, lon, geometry), group in tqdm(grouped, desc=f"{date} ", leave=False):
            try:
                result_row = compute_pv_for_group(group, date_columns=[date])
                if result_row:
                    all_results.append(result_row[0])
            except Exception as e:
                print(f"(lat={lat}, lon={lon}): {e}")
                continue

    if not all_results:
        print("no PVpot data")
        return
    all_dates = sorted(all_months_data.keys())
    columns = ["lat", "lon", "geometry"] + all_dates

    all_dfs = []
    for date in tqdm(all_months_data.keys(), desc="processing"):
        month_data = all_months_data[date]
        if not month_data:
            continue    
        dfs = []
        for var in VARIABLES:
            df = month_data[var]
            dfs.append(df)    
        try:
            merged_df = dfs[0]
            for df in dfs[1:]:
                merged_df = pd.merge(merged_df, df, on=['lon', 'lat', 'geometry'], how='inner')

            required_columns = ['lon', 'lat', 'geometry', 'u10', 'v10', 't2m', 'ssrd']
            missing = [col for col in required_columns if col not in merged_df.columns]
            if missing:
                print(f"{date} : {missing}")
                continue
            merged_df = merged_df[required_columns]
            merged_df['date'] = date
            all_dfs.append(merged_df)    
        except Exception as e:
            print(f"failed {date}: {e}")
            continue
    
    if not all_dfs:
        print("no validated data！")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)

    grouped = final_df.groupby(['lat', 'lon', 'geometry'])

    results = []
    for (lat, lon, geometry), group in tqdm(grouped, desc="calcualte PVpot"):
        meta_row = [round(float(lat), 4), round(float(lon), 4), f"POINT ({round(float(lon), 4)} {round(float(lat), 4)})"]
        pv_list = []

        for date in all_months_data.keys():
            try:
                row = group[group['date'] == date].iloc[0]
                u10 = row['u10']
                v10 = row['v10']
                t2m = row['t2m']
                ssrd = row['ssrd']
            except IndexError:
                pv_list.append(np.nan)
                continue

            try:
                u10 = float(u10)
                v10 = float(v10)
                t2m = float(t2m)
                ssrd = float(ssrd)
            except:
                pv_list.append(np.nan)
                continue

            sfcWind = np.sqrt(u10**2 + v10**2)
            tas_c = t2m - 273.15
            rsds_wm2 = ssrd / 86400.0

            T_cell = 4.3 + 0.943 * tas_c + 0.028 * rsds_wm2 - 1.528 * sfcWind
            pv = (1 - 0.005 * (T_cell - 25)) * (rsds_wm2 / 1000)
            pv_list.append(round(float(pv), 6))

        results.append(meta_row + pv_list)

    columns = ["lat", "lon", "geometry"] + list(all_months_data.keys())
    final_output_df = pd.DataFrame(results, columns=columns)
    output_path = OUTPUT_DIR / "PVpot_ERA5_2015_2024.csv"
    final_output_df.to_csv(output_path, index=False, float_format='%.6f')

    print(f"✅ PVpot：{output_path}")
    print(f"{len(results)}, {len(all_months_data)}")
    print(f" {list(all_months_data.keys())[0]} ~ {list(all_months_data.keys())[-1]}")

if __name__ == "__main__":
    main()