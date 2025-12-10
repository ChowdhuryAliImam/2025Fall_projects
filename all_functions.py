

import numpy as np
import pandas as pd
from typing import Dict, List, Literal

#  Loading & preprocessing

def aggregate_minute_to_dt(
    df: pd.DataFrame,
    dt_minutes: int,
    value_cols: List[str],
) -> pd.DataFrame:
    
    df = df.copy()

    # Check required columns 
    required_cols = {"DayID", "time_of_day_min"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"aggregate_minute_to_dt: missing required columns {missing} "
            f"in df.columns={df.columns.tolist()}"
        )

    #  Parse DayID
    df["DayID"] = pd.to_numeric(df["DayID"], errors="coerce")
    if df["DayID"].isna().all():
        raise ValueError(
            "aggregate_minute_to_dt: could not parse ANY DayID values as numeric. "
            "Check the DayID column in your CSV."
        )
    df = df.dropna(subset=["DayID"]).copy()
    df["DayID"] = df["DayID"].astype(int)
 
    tod_str = df["time_of_day_min"].astype(str)

    parsed = pd.to_datetime(tod_str, format="%H:%M:%S", errors="coerce")
    if parsed.isna().all():
        # Fallback: let pandas infer if format is a bit different
        parsed = pd.to_datetime(tod_str, errors="coerce")
        if parsed.isna().all():
            example = tod_str.dropna().iloc[0]
            raise ValueError(
                "aggregate_minute_to_dt: could not parse ANY time_of_day_min values "
                f"as times. Example value: {example!r}"
            )

    minutes_since_midnight = parsed.dt.hour * 60 + parsed.dt.minute
    df["time_of_day_min_num"] = minutes_since_midnight.astype(int)
    df = df.dropna(subset=["time_of_day_min_num"]).copy()
    if df.empty:
        raise ValueError(
            "aggregate_minute_to_dt: after parsing time_of_day_min, no rows remain. "
            "Check that column for weird values."
        )

    # slot index
    df["slot"] = (df["time_of_day_min_num"] // dt_minutes).astype(int)

    #  Aggregate 
    
    agg_dict = {col: "mean" for col in value_cols if col in df.columns}
    if not agg_dict:
        raise ValueError(
            f"aggregate_minute_to_dt: none of the requested value_cols {value_cols} "
            f"are present in df.columns={df.columns.tolist()}"
        )

    agg = (
        df.groupby(["DayID", "slot"])
        .agg(agg_dict)
        .reset_index()
    )

    if agg.empty:
        raise ValueError(
            "aggregate_minute_to_dt: aggregation produced empty DataFrame. "
            "Check dt_minutes and the range of time_of_day_min."
        )

    return agg


def load_and_aggregate_room_data(
    file_group: Dict[str, str],
    dt_minutes: int = 15,
) -> pd.DataFrame:

    # Occupancy 
    df_occ = pd.read_csv(file_group["occupancy"])
    if df_occ.empty:
        raise ValueError(
            f"load_and_aggregate_room_data: occupancy file '{file_group['occupancy']}' "
            "is empty or could not be read."
        )

    df_occ_agg = aggregate_minute_to_dt(df_occ, dt_minutes, ["N"])
    df_agg = df_occ_agg

    # --- CO2 (optional) ---
    if "co2" in file_group and file_group["co2"]:
        df_co2 = pd.read_csv(file_group["co2"])
        if not df_co2.empty:
            df_co2_agg = aggregate_minute_to_dt(df_co2, dt_minutes, ["CO2"])
            df_agg = df_agg.merge(df_co2_agg, on=["DayID", "slot"], how="left")
        else:
            print(
                f"Warning: CO2 file '{file_group['co2']}' is empty. Skipping CO2 merge."
            )

    # Damper
    if "damper" in file_group and file_group["damper"]:
        try:
            df_damper = pd.read_csv(file_group["damper"])
            if not df_damper.empty:
                df_damper_agg = aggregate_minute_to_dt(
                    df_damper, dt_minutes, ["damper"]
                )
                df_agg = df_agg.merge(df_damper_agg, on=["DayID", "slot"], how="left")
            else:
                print(
                    f"Warning: damper file '{file_group['damper']}' is empty. "
                    "Skipping damper merge."
                )
        except FileNotFoundError:
            print(
                f"Warning: damper file '{file_group['damper']}' not found. "
                "Skipping damper merge."
            )

    if df_agg.empty:
        raise ValueError(
            f"load_and_aggregate_room_data: after aggregating and merging, "
            f"no rows remain for file_group={file_group}."
        )

    return df_agg


def build_day_slices(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:

    day_slices = {
        day: g.sort_values("slot").reset_index(drop=True)
        for day, g in df.groupby("DayID")
    }
    return day_slices


def load_weather_hourly_to_dt(
    path: str,
    dt_minutes: int = 15,
) -> Dict[int, np.ndarray]:

    df = pd.read_csv(path)
    df["DayID"] = df["DayID"].astype(int)

    steps_per_hour = int(60 / dt_minutes)
    weather_slices: Dict[int, np.ndarray] = {}

    for day_id, g in df.groupby("DayID"):
        g = g.sort_values("hour_of_day")
        T_day = g["T_out"].values  # length 24
        T_expanded = np.repeat(T_day, steps_per_hour)
        weather_slices[day_id] = T_expanded

    return weather_slices



# 2. Occupancy sampling (bootstrapping)


def bootstrap_day_sequence(
    day_ids: List[int],
    n_days: int,
    rng: np.random.Generator,
) -> List[int]:
    """
    Sample a sequence of DayIDs with replacement.
    """
    return list(rng.choice(day_ids, size=n_days, replace=True))



# 3. Physics: ventilation, CO2, energy


def compute_ventilation(
    N_t: float,
    strategy: Literal["fixed", "occ_based"],
    vent_params: Dict,
) -> float:


    if strategy == "fixed":
        return vent_params["Vdot_fixed"]
    elif strategy == "occ_based":
        return vent_params["Vdot_base"] + vent_params["Vdot_per_person"] * N_t
    else:
        raise ValueError(f"Unknown strategy {strategy}")


def simulate_one_period(
    N: np.ndarray,
    T_out: np.ndarray,
    building: Dict,
    vent_params: Dict,
    config: Dict,
    strategy: Literal["fixed", "occ_based"],
) -> dict:
  
    dt = config["dt_seconds"]
    rho = 1.2   # kg/m3
    cp = 1005.0 # J/(kg*K)

    T_steps = len(N)
    C = np.zeros(T_steps + 1)   
    C[0] = building["C_out"]   

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
    day_slices: Dict[int, pd.DataFrame],
    weather_slices: Dict[int, np.ndarray],
    building: Dict,
    vent_params: Dict,
    config: Dict,
    strategy: Literal["fixed", "occ_based"],
    rng: np.random.Generator,
) -> dict:

    common_day_ids = [day_id for day_id in day_slices.keys() if day_id in weather_slices]

    if not common_day_ids:
        raise ValueError(
            "simulate_one_run: no overlapping DayID between day_slices and weather_slices. "
            f"day_slices keys = {list(day_slices.keys())}, "
            f"weather_slices keys = {list(weather_slices.keys())}"
        )

    missing_weather_days = [d for d in day_slices.keys() if d not in weather_slices]
    if missing_weather_days:
        print(
            "simulate_one_run: ignoring days with no weather data: "
            f"{missing_weather_days}"
        )

    # Sample only from days that have weather data
    sampled_days = bootstrap_day_sequence(
        common_day_ids,
        config["n_days_per_run"],
        rng,
    )

    N_list = []
    T_out_list = []

    for day_id in sampled_days:
        df_day = day_slices[day_id].sort_values("slot")
        N_day = df_day["N"].values
        T_day = weather_slices[day_id]

        if len(T_day) < len(N_day):
            raise ValueError(
                f"simulate_one_run: weather series for DayID={day_id} is shorter "
                f"than occupancy series (len(T_day)={len(T_day)}, len(N_day)={len(N_day)})."
            )

        N_list.append(N_day)
        T_out_list.append(T_day[: len(N_day)])

    N_period = np.concatenate(N_list)
    T_out_period = np.concatenate(T_out_list)

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

def run_monte_carlo(
    day_slices: Dict[int, pd.DataFrame],
    weather_slices: Dict[int, np.ndarray],
    building: Dict,
    vent_params: Dict,
    config: Dict,
    strategy: Literal["fixed", "occ_based"],
    seed: int = 42,
) -> pd.DataFrame:
 
    rng = np.random.default_rng(seed)
    results = []

    for run_idx in range(config["n_runs"]):
        metrics = simulate_one_run(
            day_slices,
            weather_slices,
            building,
            vent_params,
            config,
            strategy,
            rng,
        )
        metrics["run"] = run_idx
        metrics["strategy"] = strategy
        results.append(metrics)

    return pd.DataFrame(results)

# 4. Validation helpers
def validate_occupancy_bootstrap(
    day_slices: Dict[int, pd.DataFrame],
    config: Dict,
    n_mc_runs: int = 100,
    seed: int = 123,
) -> pd.DataFrame:
  
    original_means = [
        df_day["N"].mean() for df_day in day_slices.values()
    ]
    df_orig = pd.DataFrame(
        {"source": "original", "daily_mean_occupancy": original_means}
    )

    rng = np.random.default_rng(seed)
    all_day_ids = list(day_slices.keys())
    boot_means = []

    for _ in range(n_mc_runs * config["n_days_per_run"]):
        sampled_id = rng.choice(all_day_ids)
        df_day = day_slices[sampled_id]
        boot_means.append(df_day["N"].mean())

    df_boot = pd.DataFrame(
        {"source": "bootstrap", "daily_mean_occupancy": boot_means}
    )

    return pd.concat([df_orig, df_boot], ignore_index=True)


def validate_co2_model(
    df_day: pd.DataFrame,
    T_out_day: np.ndarray,
    building: Dict,
    vent_params: Dict,
    config: Dict,
    strategy: Literal["fixed", "occ_based"] = "fixed",
) -> pd.DataFrame:

    N = df_day["N"].values
    metrics = simulate_one_period(
        N,
        T_out_day[: len(N)],
        building,
        vent_params,
        config,
        strategy=strategy,
    )
    C_sim = metrics["C_time_series"][:-1]

    df_out = df_day.copy()
    df_out["CO2_sim"] = C_sim

    if "CO2" in df_day.columns:
        return df_out[["slot", "CO2", "CO2_sim"]]
    else:
        return df_out[["slot", "CO2_sim"]]


# 5. Hypothesis testing helpers
def run_h1_per_capita_test(df_results: pd.DataFrame) -> pd.DataFrame:
 
    df = df_results.copy()
    df = df[df["mean_occupancy"] > 0].copy()
    df["E_total_per_capita"] = df["E_total_kWh"] / df["mean_occupancy"]
    return df[["run", "mean_occupancy", "E_total_per_capita"]]


def run_h2_ventilation_test(
    df_fixed: pd.DataFrame,
    df_occ_based: pd.DataFrame,
) -> pd.DataFrame:

    dfA = df_fixed[["run", "CO2_exceed_hours", "E_total_kWh"]].copy()
    dfA["strategy"] = "fixed"

    dfB = df_occ_based[["run", "CO2_exceed_hours", "E_total_kWh"]].copy()
    dfB["strategy"] = "occ_based"

    return pd.concat([dfA, dfB], ignore_index=True)

#AI Disclosure: AI assistance was used for debugging and code suggestions in this file. Where and how will be outlined later
