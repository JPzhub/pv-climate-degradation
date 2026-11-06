# -*- coding: utf-8 -*-
"""
Daily NEX-GDDP-CMIP6 NetCDF files (SSP245/585/126, 2015-2100) are converted to CSV,
one file per variable per day. For each variable, a multi-model ensemble mean
is taken over seven CMIP6 models before the daily fields are clipped to the
province shapefile.
"""
#%%

import cartopy.crs as ccrs
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import xarray as xr
import os
import glob
import datetime

def filefolder(p):
    os.makedirs(p, exist_ok=True)

infolder=r'K:\NEX-GDDP-CMIP6-SSP245'
sec_folder = os.listdir(infolder)  
sec_folder1=sec_folder
sec_folder2=['rsds','sfcWind','tas']

inshpfile=r'K:\shp\province_proWGS84.shp'
continents = gp.read_file(inshpfile) 

outpath=r'K:\NEX-GDDP-CMIP6'
filefolder(outpath)

#%%模式集合平均计算

def moelaverage(inpath, sec):
    # Seven CMIP6 models from NEX-GDDP-CMIP6 used for the ensemble mean.
    sec_folder3 = [
        'GFDL-CM4', 'GFDL-ESM4', 'IPSL-CM6A-LR', 'MPI-ESM1-2-HR',
        'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-LM',
    ]
    # Grid label in each original .nc filename ('gr' or 'gn'); check against your files.
    grid = ['gr', 'gr', 'gr', 'gn', 'gn', 'gn', 'gn']
    ds = []
    for model, g in zip(sec_folder3, grid):
        ncpath = (inpath + '\\' + sec + '_day_' + model
                  + '_ssp245_r1i1p1f1_' + g + '_' + str(year) + '.nc')
        ds.append(xr.open_dataset(ncpath))

    # sum() aligns coords between models; all seven must share the same grid/time.
    modelAverage_result = sum(d[sec] for d in ds) / len(ds)
    return modelAverage_result


#%% 三个变量逐日导出
for sec2 in sec_folder2:
    inpath2=os.path.join(infolder,sec2)
    print('************',sec2,'************')

    for year in range(2015,2101):
        print('%s-245-%d---%d'%(sec2,year,2100))

        model_avg=moelaverage(inpath2,sec2)

        for time_i in range(0,len(model_avg.time)): #len(modelAverage_tas.time)
            
            timevalue=model_avg.time[time_i].to_pandas()
            x=timevalue.astype('M8[D]').astype('O')
            lk=x.strftime('%Y-%m-%d')
            ds_wind=model_avg[time_i]
            ds_wind.name=sec2
            edgar_wind = ds_wind.to_dataframe()  
            edgar_pv=edgar_wind        
            edgar_pv = edgar_pv.reset_index()        
            edgar_PV=gp.GeoDataFrame(edgar_pv, geometry=gp.points_from_xy(edgar_pv.lon, edgar_pv.lat)) 
            nyc_shp=continents
            edgar_gdf_solar=edgar_PV.set_crs('EPSG:4326')
               
                
            mask2_solar = gp.clip(edgar_gdf_solar, nyc_shp)
            mask2_solar.insert(0,'lon1',mask2_solar['lon'])
            mask2_solar.insert(0,'lat1',mask2_solar['lat'])
            finalout=outpath+'\\'+'SSP245-'+sec2+'-'+lk+'.csv'
            print('%s-to-csv-245' % sec2,lk,'done')
            mask2_solar.to_csv(finalout,header=True,encoding='gbk')