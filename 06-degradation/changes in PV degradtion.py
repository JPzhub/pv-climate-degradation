
# -*- coding: utf-8 -*-

import os
import re
import math
import numpy as np
import pandas as pd
from tqdm import tqdm


INPUT_DIR = r"result-PVpot"
OUTPUT_DIR = r"result-Sensitivity-ABC"

SCENARIOS = ["SSP126", "SSP245", "SSP585"]

CONFIGS = {
    "A_Baseline": {
        "lifespan": 25,
        "decay": 0.005,
        "tech_gain": 0.0
    },
    "B_Advanced": {
        "lifespan": 25,
        "decay": 0.003,
        "tech_gain": 0.002
    },
    "C_LongLife": {
        "lifespan": 30,
        "decay": 0.003,
        "tech_gain": 0.002
    }
}

BASE_YEAR = 2025
CHUNK_SIZE = 500


def get_degradation_factor(n_years, config):

    lifespan = config["lifespan"]
    replacement_cycles = n_years // lifespan
    years_in_cycle = n_years % lifespan

    degradation_factor = (
        1.0 - config["decay"]
    ) ** years_in_cycle

    technology_factor = (
        1.0 + config["tech_gain"]
    ) ** replacement_cycles

    return degradation_factor * technology_factor


def load_pvpot_data(scenario):

    filename = f"PVpot_{scenario}-2015-2100.csv"
    file_path = os.path.join(INPUT_DIR, filename)

    df = pd.read_csv(file_path, encoding="gbk")

    df = df.loc[
        :,
        ~df.columns.str.startswith("Unnamed")
    ].copy()

    required_cols = ["lat", "lon"]

    if not all(col in df.columns for col in required_cols):
        raise ValueError(
            f"Missing required columns: {required_cols}"
        )

    return df


def process_scenario(scenario, config_name, config):

    output_path = os.path.join(
        OUTPUT_DIR,
        config_name,
        scenario,
        f"{scenario}_degraded.csv"
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    df = load_pvpot_data(scenario)

    date_cols = [
        col for col in df.columns
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", col)
    ]

    if not date_cols:
        raise ValueError("No valid date columns found.")

    dates = pd.to_datetime(
        date_cols,
        format="%Y-%m-%d"
    )

    n_years = (
        dates.year.to_numpy(dtype=np.int32)
        - BASE_YEAR
    )

    factors = np.ones(
        len(n_years),
        dtype=np.float32
    )

    for i, year in enumerate(n_years):

        if year > 0:
            factors[i] = get_degradation_factor(
                year,
                config
            )

    processed_ids = set()

    if os.path.exists(output_path):

        try:
            processed_df = pd.read_csv(
                output_path,
                usecols=["grid_id"],
                dtype={"grid_id": str}
            )

            processed_ids = set(
                processed_df["grid_id"]
                .dropna()
                .str.zfill(4)
            )

            print(
                f"Resuming {config_name}-{scenario}: "
                f"{len(processed_ids)} grids already processed."
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to read existing output file: {e}"
            )

    n_grids = len(df)
    total_chunks = math.ceil(n_grids / CHUNK_SIZE)

    with tqdm(
        total=total_chunks,
        desc=f"{config_name}-{scenario}"
    ) as pbar:

        for start in range(0, n_grids, CHUNK_SIZE):

            end = min(
                start + CHUNK_SIZE,
                n_grids
            )

            chunk = df.iloc[start:end].copy()

            grid_ids = [
                f"{i:04d}"
                for i in range(start, end)
            ]

            pending_mask = [
                grid_id not in processed_ids
                for grid_id in grid_ids
            ]

            if not any(pending_mask):
                pbar.update(1)
                continue

            chunk = chunk.loc[pending_mask].copy()

            pending_ids = [
                grid_id
                for grid_id, pending in zip(
                    grid_ids, pending_mask
                )
                if pending
            ]

            data_array = chunk[date_cols].to_numpy(
                dtype=np.float32
            )

            degraded_array = (
                data_array * factors[np.newaxis, :]
            )

            output_df = pd.DataFrame(
                degraded_array,
                columns=date_cols
            )

            output_df.insert(
                0,
                "lon",
                chunk["lon"].to_numpy()
            )

            output_df.insert(
                0,
                "lat",
                chunk["lat"].to_numpy()
            )

            output_df.insert(
                0,
                "grid_id",
                pending_ids
            )

            output_df.to_csv(
                output_path,
                mode="a",
                index=False,
                header=not os.path.exists(output_path),
                encoding="utf-8-sig"
            )

            processed_ids.update(pending_ids)

            pbar.update(1)

    print(
        f"Completed {config_name}-{scenario}: "
        f"{output_path}"
    )


if __name__ == "__main__":

    for config_name, config in CONFIGS.items():

        for scenario in SCENARIOS:

            process_scenario(
                scenario,
                config_name,
                config
            )

    print("All sensitivity analyses completed.")
