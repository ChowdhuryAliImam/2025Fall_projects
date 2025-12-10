
import pandas as pd

from params import (
    SIM_CONFIG,
    BUILDING_PARAMS,
    VENTILATION_PARAMS,
    ROOM_FILE_GROUPS,
    WEATHER_FILE,
)
from all_functions import (
    load_and_aggregate_room_data,
    build_day_slices,
    load_weather_hourly_to_dt,
    run_monte_carlo,
    run_h1_per_capita_test,
    run_h2_ventilation_test,
    validate_occupancy_bootstrap,
    validate_co2_model,
)


def prepare_room_day_slices(room_name: str, dt_minutes: int) -> dict:
  
    file_group = ROOM_FILE_GROUPS[room_name]
    print(f"  Loading & aggregating data for {room_name}: {file_group}")

    df_agg = load_and_aggregate_room_data(file_group, dt_minutes=dt_minutes)

    if df_agg.empty:
        raise ValueError(
            f"prepare_room_day_slices: aggregated DataFrame is empty for {room_name}. "
            f"Check the input CSVs: {file_group}"
        )

    day_slices = build_day_slices(df_agg)

    if not day_slices:
        raise ValueError(
            f"prepare_room_day_slices: no day slices created for {room_name}. "
            "Check that 'DayID' has valid values in the aggregated data."
        )

    return day_slices


def main():
    dt_minutes = SIM_CONFIG["dt_seconds"] // 60

    weather_slices = load_weather_hourly_to_dt(
        WEATHER_FILE,
        dt_minutes=dt_minutes,
    )

    # Monte Carlo result collectors
    all_results_fixed = []
    all_results_occ_based = []

    # Validation result collectors
    all_occ_validation = []   
    all_co2_validation = []   

    for room_name in ROOM_FILE_GROUPS.keys():
        print(f"Processing {room_name}...")

       #per-room time series
 
        day_slices_room = prepare_room_day_slices(room_name, dt_minutes)

      
        # 2) Validation: Occupancy bootstrap
      
        print(f" Validating occupancy bootstrap for {room_name}...")
        df_occ_val = validate_occupancy_bootstrap(
            day_slices=day_slices_room,
            config=SIM_CONFIG,
            n_mc_runs=100,
            seed=123,
        )
        df_occ_val["room"] = room_name
        all_occ_validation.append(df_occ_val)

        # 3) Validation: CO2 model 
  
        print(f"  Validating CO2 model for {room_name}...")
        for day_id, df_day in day_slices_room.items():
            if day_id not in weather_slices:
                print(f" no weather data for DayID={day_id}, skipping.")
                continue

            T_out_day = weather_slices[day_id]
            T_out_day = T_out_day[: len(df_day)]  

            df_co2_val = validate_co2_model(
                df_day=df_day,
                T_out_day=T_out_day,
                building=BUILDING_PARAMS[room_name],
                vent_params=VENTILATION_PARAMS[room_name],
                config=SIM_CONFIG,
                strategy="occ_based",
            )
            df_co2_val["room"] = room_name
            df_co2_val["DayID"] = day_id
            all_co2_validation.append(df_co2_val)

    
        # Monte Carlo: Fixed ventilation
    
        print(f"Running Monte Carlo (fixed ventilation) for {room_name}...")
        df_fixed = run_monte_carlo(
            day_slices=day_slices_room,
            weather_slices=weather_slices,
            building=BUILDING_PARAMS[room_name],
            vent_params=VENTILATION_PARAMS[room_name],
            config=SIM_CONFIG,
            strategy="fixed",
            seed=42,
        )
        df_fixed["room"] = room_name
        all_results_fixed.append(df_fixed)

        # Monte Carlo: Occupancy-based ventilation

        print(f"  Running Monte Carlo (occ-based ventilation) for {room_name}...")
        df_occ_based = run_monte_carlo(
            day_slices=day_slices_room,
            weather_slices=weather_slices,
            building=BUILDING_PARAMS[room_name],
            vent_params=VENTILATION_PARAMS[room_name],
            config=SIM_CONFIG,
            strategy="occ_based",
            seed=43,
        )
        df_occ_based["room"] = room_name
        all_results_occ_based.append(df_occ_based)


    df_fixed_all = pd.concat(all_results_fixed, ignore_index=True)
    df_occ_based_all = pd.concat(all_results_occ_based, ignore_index=True)

    df_fixed_all.to_csv("mc_results_all_rooms_fixed.csv", index=False)
    df_occ_based_all.to_csv("mc_results_all_rooms_occ_based.csv", index=False)

    if all_occ_validation:
        df_occ_val_all = pd.concat(all_occ_validation, ignore_index=True)
        df_occ_val_all.to_csv(
            "validation_occupancy_bootstrap_all_rooms.csv",
            index=False,
        )

    if all_co2_validation:
        df_co2_val_all = pd.concat(all_co2_validation, ignore_index=True)
        df_co2_val_all.to_csv(
            "validation_co2_timeseries_all_rooms.csv",
            index=False,
        )

    df_h1_fixed = run_h1_per_capita_test(df_fixed_all)
    df_h1_fixed.to_csv("h1_per_capita_fixed.csv", index=False)

    df_h1_occ = run_h1_per_capita_test(df_occ_based_all)
    df_h1_occ.to_csv("h1_per_capita_occ_based.csv", index=False)

    df_h2 = run_h2_ventilation_test(df_fixed_all, df_occ_based_all)
    df_h2.to_csv("h2_ventilation_comparison.csv", index=False)

    print("Saved Monte Carlo, validation, and hypothesis-testing results.")


if __name__ == "__main__":
    main()

#AI Disclosure: AI assistance was used for debugging and code suggestions in this file. Where and how will be outlined later
