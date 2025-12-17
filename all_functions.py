

import numpy as np
import pandas as pd
from typing import Dict, List, Literal

#  Loading & preprocessing

def time_to_minute (time_str: str) -> int:
    """
    Converts the formatted time into minutes
    params:
    time_str: time in string format
    returns: time in minutes
    Examples:
    >>> time_to_minute("10:30:00")
    630
    >>> time_to_minute("14:15")
    855
    >>> time_to_minute("00:00")
    0
    """
    try:
        parsed = pd.to_datetime(time_str, format="%H:%M:%S")
    except ValueError:
        try:
            parsed = pd.to_datetime(time_str, format="%H:%M")
        except ValueError as e:
            raise ValueError(f"time_to_minute: invalid time format '{time_str}'. Expected 'HH:MM:SS' or 'HH:MM'.") from e
    return int(parsed.hour) * 60 + int(parsed.minute)

def aggregate_minutes (
    df: pd.DataFrame,
    value_col:str,
    dt_minutes:int= 15,
    day_col: str = "DayID",
    time_col: str = "time_of_day_min"
) -> pd.DataFrame:
    """
    Aggregates data in a DataFrame by day and time slot
    params:
    df: DataFrame (from the daataset of occupancy)
    value_col: column name of value to aggregate
    dt_minutes: number of minutes to aggregate
    day_col: column name of day to aggregate
    time_col: column name of time slot to aggregate
    returns: DataFrame (aggregated data)
    Examples:
    Assuming a sample DataFrame df with columns 'DayID', 'time_of_day_min' and 'N' where N is a value to aggregate.
    >>> import pandas as pd
    >>> data = {'DayID': [1, 1, 1, 1, 2, 2, 2, 2],
    ...         'time_of_day_min': ['00:00:00', '00:15:00', '00:30:00', '00:45:00', '01:00:00', '01:15:00', '01:30:00', '01:45:00'],
    ...         'N': [10, 12, 14, 16, 20, 22, 24, 26]}
    >>> df = pd.DataFrame(data)
    >>> aggregated_df = aggregate_minutes(df, 'N', dt_minutes=15)
    >>> set(aggregated_df.columns) == {'DayID', 'slot', 'N'}
    True
    >>> len(aggregated_df) == 8
    True
    >>> aggregated_df['N'].iloc[0] == 10
    True
    >>> aggregated_df['N'].iloc[1] == 12
    True
    """
    df = df.copy()
    df["DayID"] = df[day_col].astype(int)
    df["time_of_day_min_num"] = df[time_col].apply(time_to_minute).astype(int)
    df["slot"] = (df["time_of_day_min_num"] // dt_minutes).astype(int)
    agg_df = df.groupby(["DayID", "slot"], as_index=False)[value_col].mean()
    return agg_df


def load_and_aggregate_room_data(
    file_group: Dict[str, str],
    dt_minutes: int = 15,
) -> pd.DataFrame:
    """
    Loads and aggregates room data from occupancy and CO2 CSV files.
    params:
    file_group: Dictionary of file names to file paths. in the params file the rooms are grouped, and this function will take a room group and aggregate the data
    dt_minutes: number of minutes to aggregate
    returns: DataFrame (aggregated data)

    Examples:
    >>> import tempfile
    >>> import os
    >>> import pandas as pd
    >>> with tempfile.TemporaryDirectory() as tmpdirname:
    ...     # Create dummy occupancy data
    ...     occupancy_data = {'DayID': [1, 1, 1, 1], 'time_of_day_min': ['00:00:00', '00:15:00', '00:30:00', '00:45:00'], 'N': [10, 12, 14, 16]}
    ...     occupancy_df = pd.DataFrame(occupancy_data)
    ...     occupancy_file = os.path.join(tmpdirname, 'occupancy.csv')
    ...     occupancy_df.to_csv(occupancy_file, index=False)
    ...     file_group = {"occupancy": occupancy_file}
    ...     df_agg = load_and_aggregate_room_data(file_group, dt_minutes=15)
    ...     set(df_agg.columns) == {'DayID', 'slot', 'N'}
    True
    >>> len(df_agg)
    4
    >>> int( df_agg['N'].iloc[0])
    10
    """
    # Occupancy 
    df_occ = pd.read_csv(file_group["occupancy"])
    df_occ_agg = aggregate_minutes(df_occ, "N", dt_minutes)
    df_agg = df_occ_agg

    return df_agg

def sample_day_ids(df: pd.DataFrame, n_days: int = 44, replace: bool = True) -> np.ndarray:
    """
    Samples a specified number of unique DayIDs from a DataFrame.
    params:
    df: DataFrame
    n_days: number of days to sample
    replace: whether to replace any existing DayIDs
    returns: array of DayIDs

    Examples:
    >>> data = {'DayID': [1, 1, 2, 2, 3, 3]}
    >>> df = pd.DataFrame(data)
    >>> sampled_days = sample_day_ids(df, n_days=2, replace=False)
    >>> len(sampled_days) <= 3
    True
    >>> sampled_days.size == 2
    True
    >>> sampled_days = sample_day_ids(df, n_days=5, replace=True)
    >>> len(sampled_days) == 5
    True
    """
    day_ids = df["DayID"].unique()
    sampled_days = np.random.choice(day_ids, size=n_days, replace=replace)
    return sampled_days

def build_44day_from_ids(df: pd.DataFrame, sampled_days: np.ndarray) -> pd.DataFrame:
    """
    Selects and concatenates data from a DataFrame based on a list of DayIDs.
    PARAMS:
    df: DataFrame
    sampled_days: array of DayIDs
    returns: DataFrame

    Examples:
    >>> data = {'DayID': [1, 1, 2, 2, 3, 3], 'slot': [0, 1, 0, 1, 0, 1], 'value': [10, 11, 20, 21, 30, 31]}
    >>> df = pd.DataFrame(data)
    >>> sampled_days = np.array([1, 3])
    >>> df_44 = build_44day_from_ids(df, sampled_days)
    >>> set(df_44['DayID'].unique()) == {1, 3}
    True
    >>> len(df_44) == 4
    True
    >>> df_44['value'].iloc[0] == 10
    True
    """
    df_sorted = df.sort_values(["DayID", "slot"]).reset_index(drop=True)
    groups = df_sorted.groupby("DayID")

    slices = [groups.get_group(d) for d in sampled_days]
    df_44 = pd.concat(slices, ignore_index=True)
    return df_44


def load_weather_hourly_to_dt(
    path: str,
    dt_minutes: int = 15,
    day_col: str = "DayID",
    time_col: str = "hour_of_day",
    outdoor_temp_col: str = "T_out"
) -> pd.DataFrame:
    """
    Loads weather data from a CSV, converts the time to minutes, and resamples to a finer time resolution.
    Params:
    path: path to weather file
    dt_minutes: time interval in minutes
    day_col: name of day column
    time_col: name of time column
    outdoor_temp_col: name of outdoor temperature column
    returns: DataFrame

    Examples:
    >>> import tempfile
    >>> import os
    >>> with tempfile.TemporaryDirectory() as tmpdirname:
    ...     # Create dummy weather data
    ...     weather_data = {'DayID': [1, 1], 'hour_of_day': ['01:00', '02:00'], 'T_out': [10, 12]}
    ...     weather_df = pd.DataFrame(weather_data)
    ...     weather_file = os.path.join(tmpdirname, 'weather.csv')
    ...     weather_df.to_csv(weather_file, index=False)
    ...     df_resampled = load_weather_hourly_to_dt(weather_file, dt_minutes=30)
    ...     len(df_resampled) == 4
    ...
    True
    >>> df_resampled['T_out'].iloc[0] == 10.0
    True
    >>> df_resampled['T_out'].iloc[1] == 10.0
    True
    >>> df_resampled['T_out'].iloc[2] == 12.0
    True
    >>> df_resampled['T_out'].iloc[3] == 12.0
    True
    """
    df = pd.read_csv(path, header=0)

    if 60 % dt_minutes != 0:
        raise ValueError("dt_minutes must divide 60 exactly.")
    
    steps_per_hour = int(60 / dt_minutes)
    df[day_col] = df[day_col].astype(int)
    df["minutes"] = df[time_col].apply(time_to_minute)
    df = df[[day_col, "minutes", outdoor_temp_col]].sort_values([day_col, "minutes"]).reset_index(drop=True)

    repeated_index = df.index.repeat(steps_per_hour)
    repeated_df = df.loc[repeated_index].copy()

    additions = np.tile(np.arange(0, 60, dt_minutes), len(df))
    repeated_df["minutes"] = (repeated_df["minutes"].to_numpy() + additions).astype(int)
    repeated_df["slot"] = (repeated_df["minutes"] // dt_minutes).astype(int)
    repeated_df = repeated_df.sort_values([day_col, "slot"]).reset_index(drop=True)
    return repeated_df.rename(columns={day_col: "DayID", outdoor_temp_col: "T_out"})


def attach_weather_to_room(df_room: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    """
    Merges room data with weather data based on 'DayID' and 'slot'.
    Params:
    df_room: DataFrame of room data
    df_weather: DataFrame of weather data
    returns: DataFrame

    Examples:
    >>> room_data = {'DayID': [1, 1], 'slot': [0, 1], 'N': [10, 12]}
    >>> room_df = pd.DataFrame(room_data)
    >>> weather_data = {'DayID': [1, 1], 'slot': [0, 1], 'T_out': [10, 12]}
    >>> weather_df = pd.DataFrame(weather_data)
    >>> merged_df = attach_weather_to_room(room_df, weather_df)
    >>> 'T_out' in merged_df.columns
    True
    >>> merged_df['T_out'].iloc[0] == 10
    True
    >>> merged_df['T_out'].iloc[1] == 12
    True
    """
    w = df_weather[["DayID", "slot", "T_out"]].copy()
    out = df_room.merge(w, on=["DayID", "slot"], how="left")
    if out["T_out"].isna().any():
        raise ValueError("Missing T_out after merge; check DayID/slot alignment.")
    return out

#  Physics: ventilation, CO2, energy
def compute_ventilation(
    N_t: float,
    strategy: Literal["fixed", "occ_based"],
    vent_params: Dict,
) -> float:
    """
    Computes the ventilation rate based on the chosen strategy.
    params:
    N_t: number of occupants at time t
    strategy: name of ventilation strategy
    returns: computed ventilation measure

    Examples:
    >>> vent_params = {"Vdot_fixed": 100, "Vdot_base": 50, "Vdot_per_person": 10}
    >>> compute_ventilation(10, "fixed", vent_params)
    100
    >>> compute_ventilation(10, "occ_based", vent_params)
    150
    """
    if strategy == "fixed":
        return vent_params["Vdot_fixed"]
    elif strategy == "occ_based":
        return vent_params["Vdot_base"] + vent_params["Vdot_per_person"] * N_t
    else:
        raise ValueError(f"Unknown strategy {strategy}")

def transform_occupancy(
    N: np.ndarray,
    multiplier: float = 1.0,
    cap:float = None
) -> np.ndarray:
    """
    Increase/decrease occupancy by a multiplier (shifts mean), with optional cap.
    Returns float array (so downstream math is safe).
    """
    N2 = np.round(N.astype(float) * multiplier)
    if cap is not None:
        N2 = np.clip(N2, 0, cap)
    else:
        N2 = np.clip(N2, 0, np.inf)
    return N2.astype(float)

def simulate_one_period(
    N: np.ndarray,
    T_out: np.ndarray,
    building: Dict,
    vent_params: Dict,
    config: Dict,
    strategy: Literal["fixed", "occ_based"],
    rho: float = 1.2,
    cp: float = 1005.0,
) -> dict:
    """
    Simulates one period of building operation, calculating energy consumption and CO2 levels.
    Params:
    N: array of total items in the period
    T_out: outdoor temperature
    building: building parameters dictionary
    vent_params: vent parameters dictionary
    config: configuration dictionary
    strategy: name of ventilation strategy
    rho: efficiency parameter
    cp: capacity parameter
    returns: dict
    Examples:
    >>> import numpy as np
    >>> building = {"V": 461.48, "UA": 150.0, "T_set": 22.0, "eta_heat": 0.95, "COP_cool": 3.0, "q_person": 75.0, "G_CO2": 0.000005, "C_out": 420.0, "P_plug": 500.0, "P_light": 400.0}
    >>> vent_params = {"Vdot_fixed": 0.923, "Vdot_base": 0.0834, "Vdot_per_person": 0.01}
    >>> config = {"dt_seconds": 900, "co2_threshold": 1000.0, "n_days_per_run": 6.0, "n_runs": 100}
    >>> N = np.array([1, 1, 1, 1, 1, 1])
    >>> T_out = np.array([10, 10, 10, 10, 10, 10])
    >>> results = simulate_one_period(N, T_out, building, vent_params, config, strategy= 'fixed')
    >>> "E_heat_kWh" in results
    True
    >>> "E_cool_kWh" in results
    True
    >>> "E_plug_kWh" in results
    True
    >>> "E_light_kWh" in results
    True
    >>> "E_total_kWh" in results
    True
    >>> "CO2_exceed_hours" in results
    True
    """
    dt = config["dt_seconds"]
    T_steps = len(N)
    C = np.zeros(T_steps + 1)   
    C[0] = building['C_out']   

    Q_heat = np.zeros(T_steps)
    Q_cool = np.zeros(T_steps)
    E_plug = np.zeros(T_steps)
    E_light = np.zeros(T_steps)

    for t in range(T_steps):
        N_t = N[t]
        T_out_t = T_out[t]

        Vdot_t = compute_ventilation(N_t, strategy, vent_params)

        # Internal gains
        Q_int = N_t * building["q_person"]

        if N_t > 0:
            Q_plug_t = building["P_plug"]
            Q_light_t = building["P_light"]
        else:
            Q_plug_t = 0.0
            Q_light_t = 0.0

        E_plug[t] = Q_plug_t * dt
        E_light[t] = Q_light_t * dt

        # Envelope load
        Q_env = building["UA"] * (building["T_set"] - T_out_t)

        # Ventilation load
        m_dot = rho * Vdot_t
        Q_vent = m_dot * cp * (building["T_set"] - T_out_t)

        # Net heating / cooling (W)
        Q_total = Q_env + Q_vent - Q_int

        if Q_total > 0:
            Q_heat[t] = Q_total
        else:
            Q_cool[t] = -Q_total
       
        C[t + 1] = C[t] + dt / building["V"] * (
            building["G_CO2"] * N_t + Vdot_t * (building["C_out"] - C[t])
        )

    # Convert to energy in kWh
    J_to_kWh = 1.0 / 3.6e6

    E_heat_J = np.sum(Q_heat * dt) / building["eta_heat"]
    E_cool_J = np.sum(Q_cool * dt) / building["COP_cool"]
    E_plug_J = np.sum(E_plug)
    E_light_J = np.sum(E_light)

    metrics = {
        "E_heat_kWh": E_heat_J * J_to_kWh,
        "E_cool_kWh": E_cool_J * J_to_kWh,
        "E_plug_kWh": E_plug_J * J_to_kWh,
        "E_light_kWh": E_light_J * J_to_kWh,
        "E_total_kWh": (E_heat_J + E_cool_J + E_plug_J + E_light_J) * J_to_kWh,
        "C_time_series": C,
    }

    # CO2 exceedance hours
    exceed = C[:-1] > config["co2_threshold"]
    exceed_minutes = exceed.sum() * (dt / 60.0)
    metrics["CO2_exceed_hours"] = exceed_minutes / 60.0

    return metrics

def simulate_one_run(
    df_44: pd.DataFrame,
    building: dict,
    vent_params: dict,
    config: dict,
    strategy: Literal["fixed", "occ_based"],
) -> dict:
    """
    Simulates a single run of the model, calculating energy consumption and CO2 metrics.
    Parameters:
    df_44: pandas DataFrame
    building: dict of building parameters
    vent_params: dict of vent parameters
    config: dict of configuration parameters
    strategy: string of ventilation strategy
    Returns: dict of metrics

    Examples:
    >>> import pandas as pd
    >>> import numpy as np
    >>> building = {"V": 461.48, "UA": 150.0, "T_set": 22.0, "eta_heat": 0.95, "COP_cool": 3.0, "q_person": 75.0, "G_CO2": 0.000005, "C_out": 420.0, "P_plug": 500.0, "P_light": 400.0}
    >>> vent_params = {"Vdot_fixed": 0.923, "Vdot_base": 0.0834, "Vdot_per_person": 0.01}
    >>> config = {"dt_seconds": 900, "co2_threshold": 1000.0, "n_days_per_run": 6.0, "n_runs": 100}
    >>> df_44 = pd.DataFrame({"N": [10, 10, 10, 0, 0, 0], "T_out": [10, 10, 10, 10, 10, 10]})
    >>> results = simulate_one_run(df_44, building, vent_params, config, "fixed")
    >>> "mean_occupancy" in results
    True
    >>> "E_heat_kWh" in results
    True
    >>> results["mean_occupancy"] == 5.0
    True
    """
    N_period = df_44["N"].to_numpy()
    T_out_period = df_44["T_out"].to_numpy()

    metrics = simulate_one_period(
        N_period,
        T_out_period,
        building,
        vent_params,
        config,
        strategy,
    )
    metrics["mean_occupancy"] = float(N_period.mean())
    return metrics


def run_room_mc_with_samples(
    df_room_with_weather: pd.DataFrame,
    building: Dict,
    vent_params: Dict,
    config: Dict,
    strategy: Literal["fixed", "occ_based"],
    sampled_days_list: List[np.ndarray],
    occ_multiplier: float = 1.0,
    occ_cap: float = None,
) -> pd.DataFrame:
    """
    Run MC using a pre-generated list of sampled DayID sequences.
    This makes baseline vs treatment PAIRABLE (same sampled days per run).

    Run MC using a pre-generated list of sampled DayID sequences.
    This makes baseline vs treatment PAIRABLE (same sampled days per run).

    - df_room_with_weather: must contain columns DayID, slot, N, T_out
    - sampled_days_list: list of arrays, each array length = n_days_per_run (e.g., 44)

    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     "DayID": np.repeat(np.arange(10), 24),
    ...     "slot": np.tile(np.arange(24), 10),
    ...     "N": np.random.randint(0, 10, 240),
    ...     "T_out": np.random.rand(240),
    ... })
    >>> building = {"V": 461.48, "UA": 150.0, "T_set": 22.0, "eta_heat": 0.95, "COP_cool": 3.0, "q_person": 75.0, "G_CO2": 0.000005, "C_out": 420.0, "P_plug": 500.0, "P_light": 400.0}
    >>> vent_params = {"Vdot_fixed": 0.923, "Vdot_base": 0.0834, "Vdot_per_person": 0.01}
    >>> config = {"dt_seconds": 900, "co2_threshold": 1000.0, "n_days_per_run": 6.0, "n_runs": 100}
    >>> strategy = "fixed"
    >>> sampled_days_list = [np.arange(0, 44) % 10]  # One run with 44 days (wrapping around)
    >>> result = run_room_mc_with_samples(df, building, vent_params, config, strategy, sampled_days_list, occ_multiplier=2.0, occ_cap=5.0)
    >>> result.shape[0]  # Number of runs
    1
    >>> (result['E_heat_kWh'] > 100).item() # CO2 avg should be above zero.
    True
    """
    rows = []

    for run_idx, sampled_days in enumerate(sampled_days_list):
        df_44 = build_44day_from_ids(df_room_with_weather, sampled_days)

        # Apply occupancy intervention (H1 treatment)
        df_44 = df_44.copy()
        df_44["N"] = transform_occupancy(
            df_44["N"].to_numpy(),
            multiplier=occ_multiplier,
            cap=occ_cap
        )

        metrics = simulate_one_run(
            df_44=df_44,
            building=building,
            vent_params=vent_params,
            config=config,
            strategy=strategy,
        )

        metrics["run"] = run_idx
        rows.append(metrics)

    return pd.DataFrame(rows)



#Validation helpers
def validate_room1_occupancy_sampling(
    df_room1: pd.DataFrame,
    n_runs: int = 10,
    n_days: int = 44,
) -> pd.DataFrame:
    """
    Bootstrap occupancy for Room 1 and return a combined DataFrame
    for statistical validation.
    """
    occ_runs = []

    for _ in range(n_runs):
        sampled_day_ids = sample_day_ids(df_room1, n_days=n_days, replace=True)
        df_44 = build_44day_from_ids(df_room1, sampled_day_ids)
        occ_runs.append(df_44[["N"]])

    combined_df = pd.concat(occ_runs, ignore_index=True)
    return combined_df


def validate_room1_co2_sampling(
    df_room1: pd.DataFrame,
    building: Dict,
    vent_params: Dict,
    config: Dict,
    strategy: Literal["fixed", "occ_based"],
    n_runs: int = 10,
    n_days: int = 44,
) -> pd.DataFrame:
    co2_runs = []

    for _ in range(n_runs):
        sampled_day_ids = sample_day_ids(df_room1, n_days=n_days, replace=True)
        df_44 = build_44day_from_ids(df_room1, sampled_day_ids)

        metrics = simulate_one_run(df_44, building, vent_params, config, strategy)
        C_series = metrics["C_time_series"][:-1]
        co2_runs.append(pd.DataFrame({"CO2_sim": C_series}))

    return pd.concat(co2_runs, ignore_index=True)


# Hypothesis testing helpers
def run_h1_per_capita_test(df_results: pd.DataFrame) -> pd.DataFrame:

    """
    Computes energy consumption per capita.
    Examples:
    >>> df_results = pd.DataFrame({"run": [0, 1, 2], "mean_occupancy": [10, 0, 20], "E_total_kWh": [100, 50, 200]})
    >>> results = run_h1_per_capita_test(df_results)
    >>> "E_total_per_capita" in results.columns
    True
    >>> len(results) == 2
    True
    >>> results["E_total_per_capita"].iloc[0] == 10.0
    True
    """
 
    df = df_results.copy()
    df = df[df["mean_occupancy"] > 0].copy()
    df["E_total_per_capita"] = df["E_total_kWh"] / df["mean_occupancy"]
    return df[["run", "mean_occupancy", "E_total_per_capita"]]

def run_h2_ventilation_test(
    df_fixed: pd.DataFrame,
    df_occ_based: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combines results from fixed and occupancy-based ventilation strategies.

    Examples:
    >>> fixed_data = {'run': [0, 1], 'CO2_exceed_hours': [1, 2], 'E_total_kWh': [100, 200]}
    >>> occ_based_data = {'run': [0, 1], 'CO2_exceed_hours': [3, 4], 'E_total_kWh': [300, 400]}
    >>> df_fixed = pd.DataFrame(fixed_data)
    >>> df_occ_based = pd.DataFrame(occ_based_data)
    >>> combined_df = run_h2_ventilation_test(df_fixed, df_occ_based)
    >>> 'strategy' in combined_df.columns
    True
    >>> len(combined_df) == 4
    True
    >>> combined_df['strategy'].iloc[0] == 'fixed'
    True
    >>> combined_df['CO2_exceed_hours'].iloc[2] == 3
    True
    """

    dfA = df_fixed[["run", "CO2_exceed_hours", "E_total_kWh"]].copy()
    dfA["strategy"] = "fixed"

    dfB = df_occ_based[["run", "CO2_exceed_hours", "E_total_kWh"]].copy()
    dfB["strategy"] = "occ_based"

    return pd.concat([dfA, dfB], ignore_index=True)



if __name__ == "__main__":
    main()
#AI Disclosure: ChatGPT assistance was used for debugging and code suggestions in this project. Google AI studio was used for doctest help.