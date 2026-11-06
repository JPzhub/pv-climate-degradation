# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import norm


data_dir = r"result-PVpot"
output_dir = r"result-PVpot tend"
os.makedirs(output_dir, exist_ok=True)
SCENARIOS = ['SSP126', 'SSP245', 'SSP585']
variables = ['PVpot']

def load_data(scenario):

    filename = f"PVpot_{scenario}-2015-2100.csv"
    file_path = os.path.join(data_dir, filename)    
    try:
        df = pd.read_csv(file_path)
        df = df[[col for col in df.columns if not col.startswith('Unnamed')]]

        required_cols = ['lat', 'lon']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"{required_cols}")
        if 'geometry' not in df.columns:
            df['geometry'] = df.apply(lambda row: f"POINT ({row['lon']} {row['lat']})", axis=1)

        date_cols = [col for col in df.columns if '-' in col and len(col) == 10 and col[4] == '-' and col[7] == '-']
        if not date_cols:
            raise ValueError("no validated date")

        data = df.set_index(['lat', 'lon', 'geometry'])[date_cols].T
        data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
        return data
    except Exception as e:
        return None

def sens_slope_mk_test(data):
    """
   Sen's slope and Mann-Kendall test
    """
    n = len(data)
    slopes = []

    for i in range(n):
        for j in range(i+1, n):
            if j == i: continue
            slope = (data[j] - data[i]) / (j - i)
            slopes.append(slope)

    sens_slope = np.median(slopes) * 10 if slopes else 0
    
    # Mann-Kendall test
    s = 0
    for i in range(n-1):
        for j in range(i+1, n):
            s += np.sign(data[j] - data[i])

    var_s = (n*(n-1)*(2*n+5)) / 18
    if n < 10:
        var_s -= sum(k*(k-1)*(2*k+5) for k in (np.sum(data == x) for x in set(data)))

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    
    p = 2 * (1 - norm.cdf(abs(z)))
    return sens_slope, z, p


def calculate_grid_trend(scenario):

    full_data = load_data(scenario)
    if full_data is None:
        return None, None

    future = full_data.loc['2026-01-01':'2100-12-31']
    hist = full_data.loc['2015-01-01':'2025-12-31']

    hist_annual = hist.resample('Y').sum()
    hist_mean = hist_annual.mean(axis=0)

    annual_yield = future.resample('Y').sum()

    grid_trends = []
    grid_changes = []

    for grid_point in tqdm(annual_yield.columns, desc="processing"):

        ts = annual_yield[grid_point].values
        slope, z, p = sens_slope_mk_test(ts)

        grid_trends.append({
            'lat': grid_point[0],
            'lon': grid_point[1],
            'geometry': grid_point[2],
            'Sen(W/decade)': round(slope, 4),
            'MK-Z': round(z, 2),
            'MK-p': round(p, 4),
            'sig': 'up' if p < 0.05 and slope > 0 else 
                       'down' if p < 0.05 and slope < 0 else 
                       'none'
        })       

        hist_avg = hist_mean[grid_point]
        future_avg = annual_yield[grid_point].mean()
        if abs(hist_avg) > 1e-6:
            change = (future_avg - hist_avg) / hist_avg * 100
        else:
            change = np.nan
        grid_changes.append({
            'lat': grid_point[0],
            'lon': grid_point[1],
            'geometry': grid_point[2],
            'hisMean(W)': round(hist_avg, 2),
            'futureMean(W)': round(future_avg, 2),
            'rate(%)': round(change, 2)
        })
    
    return pd.DataFrame(grid_trends), pd.DataFrame(grid_changes)

if __name__ == "__main__":
    for scenario in SCENARIOS:
        print(f"{scenario}")
        
        grid_trend, grid_change = calculate_grid_trend(scenario)
        
        if grid_trend is None or grid_change is None:

            continue
        
        output_path = os.path.join(output_dir, f"{scenario}_PVpot.xlsx")
        
        with pd.ExcelWriter(output_path) as writer:
            grid_trend.to_excel(writer, sheet_name="trend", index=False)
            grid_change.to_excel(writer, sheet_name="rate", index=False)
        
        print(f"{scenario}：{output_path}")
    
    print("done！")