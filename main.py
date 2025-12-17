import numpy as np
import pandas as pd
import params
from all_functions import  *


def compute_distribution_stats(df: pd.DataFrame) -> pd.DataFrame:
    """mean, std, min, max, skewness, kurtosis for numeric columns."""
    stats = df.describe().T
    stats["skewness"] = df.skew(numeric_only=True)
    stats["kurtosis"] = df.kurtosis(numeric_only=True)
    return stats[["mean", "std", "min", "max", "skewness", "kurtosis"]]


def main():

    # Load + merge Room1 data

    file_group = params.ROOM_FILE_GROUPS["Room1"]
    df_room1 = load_and_aggregate_room_data(file_group, dt_minutes=15)
    df_weather = load_weather_hourly_to_dt(params.WEATHER_FILE, dt_minutes=15)
    df_room1_w = attach_weather_to_room(df_room1, df_weather)

    # Validation 1: Occupancy bootstrap (10 runs)

    occ_df = validate_room1_occupancy_sampling(
        df_room1=df_room1,
        n_runs=10,
        n_days=44,
    )
    print("\n=== Validation 1: Room1 Occupancy Bootstrap Stats (10×44 days) ===")
    print(compute_distribution_stats(occ_df))

    # Validation 2: CO2 simulation sampling (10 runs)
    co2_df = validate_room1_co2_sampling(df_room1_w, building=params.BUILDING_PARAMS["Room1"],vent_params=params.VENTILATION_PARAMS["Room1"], config=params.SIM_CONFIG,strategy="occ_based")

    print("\n=== Validation 2: Room1 CO2 Simulation Stats (10×44 days) ===")
    print(compute_distribution_stats(co2_df))


    # Monte Carlo runs for hypotheses (fixed vs occ_based)

    rooms = ["Room1", "Room2", "Room3"]
    cap_map = {"Room1": 84, "Room2": 32, "Room3": 32}
    df_weather = load_weather_hourly_to_dt(params.WEATHER_FILE, dt_minutes=15)
    # Load + attach weather per room
    room_df = {}
    for r in rooms:
        df_r = load_and_aggregate_room_data(params.ROOM_FILE_GROUPS[r], dt_minutes=15)
        df_rw = attach_weather_to_room(df_r, df_weather)
        room_df[r] = df_rw

    n_runs = int(params.SIM_CONFIG["n_runs"])
    n_days = int(params.SIM_CONFIG["n_days_per_run"])

    # Pre-sample day sequences
    sampled_days_by_room = {
        r: [sample_day_ids(room_df[r], n_days=n_days, replace=True) for _ in range(n_runs)]
        for r in rooms
    }

    sum_cols = ["E_heat_kWh", "E_cool_kWh", "E_plug_kWh", "E_light_kWh",
                "E_total_kWh", "CO2_exceed_hours", "mean_occupancy"]

    # H1: baseline vs increased occupancy (FIXED ventilation)

    mult = 1.50  # example +20%

    base_room_results = []
    treat_room_results = []

    for r in rooms:
        base_room_results.append(
            run_room_mc_with_samples(
                df_room_with_weather=room_df[r],
                building=params.BUILDING_PARAMS[r],
                vent_params=params.VENTILATION_PARAMS[r],
                config=params.SIM_CONFIG,
                strategy="fixed",
                sampled_days_list=sampled_days_by_room[r],
                occ_multiplier=1.0,
                occ_cap=cap_map[r],
            )
        )
        treat_room_results.append(
            run_room_mc_with_samples(
                df_room_with_weather=room_df[r],
                building=params.BUILDING_PARAMS[r],
                vent_params=params.VENTILATION_PARAMS[r],
                config=params.SIM_CONFIG,
                strategy="fixed",
                sampled_days_list=sampled_days_by_room[r],
                occ_multiplier=mult,
                occ_cap=cap_map[r],
            )
        )

    df_base = pd.concat(base_room_results, ignore_index=True)
    df_treat = pd.concat(treat_room_results, ignore_index=True)

    base_tot = df_base.groupby("run", as_index=False)[sum_cols].sum()
    treat_tot = df_treat.groupby("run", as_index=False)[sum_cols].sum()

    base_tot["E_per_capita"] = base_tot["E_total_kWh"] / base_tot["mean_occupancy"]
    treat_tot["E_per_capita"] = treat_tot["E_total_kWh"] / treat_tot["mean_occupancy"]

    h1_compare = pd.DataFrame({
    "run": base_tot["run"],
    "E_per_capita_baseline": base_tot["E_per_capita"].to_numpy(),
    "E_per_capita_treatment": treat_tot["E_per_capita"].to_numpy(),
})


    print("\nSummary (baseline):")
    print(h1_compare["E_per_capita_baseline"].describe(percentiles=[0.05, 0.5, 0.95]))

    print("\nSummary (treatment):")
    print(h1_compare["E_per_capita_treatment"].describe(percentiles=[0.05, 0.5, 0.95]))


    diff = treat_tot["E_per_capita"] - base_tot["E_per_capita"]

    print("\n=== H1 (Building total, FIXED): Increased occupancy vs baseline ===")
    print(f"Occupancy multiplier = {mult} (capped by seating)")
    print(diff.describe(percentiles=[0.05, 0.5, 0.95]))


    # H2: fixed vs occ_based (baseline occupancy)

    fixed_room_results = []
    occ_room_results = []

    for r in rooms:
        fixed_room_results.append(
            run_room_mc_with_samples(
                df_room_with_weather=room_df[r],
                building=params.BUILDING_PARAMS[r],
                vent_params=params.VENTILATION_PARAMS[r],
                config=params.SIM_CONFIG,
                strategy="fixed",
                sampled_days_list=sampled_days_by_room[r],
                occ_multiplier=1.0,
                occ_cap=cap_map[r],
            )
        )
        occ_room_results.append(
            run_room_mc_with_samples(
                df_room_with_weather=room_df[r],
                building=params.BUILDING_PARAMS[r],
                vent_params=params.VENTILATION_PARAMS[r],
                config=params.SIM_CONFIG,
                strategy="occ_based",
                sampled_days_list=sampled_days_by_room[r],
                occ_multiplier=1.0,
                occ_cap=cap_map[r],
            )
        )

    df_fixed = pd.concat(fixed_room_results, ignore_index=True)
    df_occ = pd.concat(occ_room_results, ignore_index=True)

    fixed_tot = df_fixed.groupby("run", as_index=False)[sum_cols].sum()
    occ_tot = df_occ.groupby("run", as_index=False)[sum_cols].sum()

    fixed_tot["strategy"] = "fixed"
    occ_tot["strategy"] = "occ_based"

    h2_tot = pd.concat([fixed_tot, occ_tot], ignore_index=True)

    print("\n=== H2 (Building total): Fixed vs Occ-based ===")
    print(h2_tot.groupby("strategy")[["CO2_exceed_hours", "E_total_kWh"]].describe())

    # Save only totals (as you requested)
    base_tot.to_csv("H1_baseline_building.csv", index=False)
    treat_tot.to_csv("H1_treatment_building.csv", index=False)
    h2_tot.to_csv("H2_building_totals.csv", index=False)
    print("\nSaved: H1_baseline_building.csv, H1_treatment_building.csv, H2_building_totals.csv")


if __name__ == "__main__":
    main()

#AI Disclosure: ChatGPT assistance was used for debugging and code suggestions in this project. Google AI studio was used for doctest help.