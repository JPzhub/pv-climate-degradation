# pv-climate-degradation

Supporting code for the paper:

**Climate change decouples solar resource advantage from photovoltaic resilience in China**

This repository contains the majority of the computational workflows used in the study.

## Repository Structure

- **01 PV Potential** – Calculation of photovoltaic potential
- **02-Trend analysis** – Temporal trend analysis
- **03-PVpot CV** – Coefficient of variation of PV potential
- **04-Extreme weather** – Extreme weather impact analysis
- **05-Climate-driven Photovoltaic potential response and risk metrics** – Climate-driven response and risk metrics
- **06-degradation** – PV system degradation modeling under different technological strategies
- **07-CMDl and CDSI** – Calculation of Climate-Modulated Degradation Impact (CMDI) and Climate-Degradation Sensitivity Index (CDSI) with SHAP-based attribution

## Requirements

- Python >= 3.9

Install dependencies:

```bash
pip install -r requirements.txt
```

## Main packages

```text
numpy
pandas
dask
scikit-learn
shap
joblib
tqdm
pyarrow
scipy
matplotlib
xarray
netCDF4
```

## Data Availability

The climate inputs used in this study are publicly available third-party products and are therefore not redistributed in this repository:

- **Future climate projections** — NASA Earth Exchange Global Daily Downscaled Projections (NEX-GDDP-CMIP6), version, SSP1-2.6, SSP2-4.5 and SSP5-8.5, seven CMIP6 models (GFDL-CM4, GFDL-ESM4, IPSL-CM6A-LR, MPI-ESM1-2-HR, MPI-ESM1-2-LR, MRI-ESM2-0, NorESM2-LM), variables `tas`, `rsds` and `sfcWind`, 2026–2100, daily, 0.25° x 0.25°. Available from the NASA NCCS archive at https://www.nccs.nasa.gov/data-collections/nex-gddp-cmip6/ and described in Thrasher et al. (2022), https://doi.org/10.1038/s41597-022-01393-4.
- **Historical reference fields** — ERA5 hourly single-level reanalysis, 2015–2025, aggregated to daily. Available from the Copernicus Climate Data Store at https://doi.org/10.24381/cds.adbb2d47; see Hersbach et al. (2020), https://doi.org/10.1002/qj.3803.


## Usage

Each major analysis step is organized in its own folder.
Please refer to the scripts and comments inside the individual folders for specific usage instructions.
