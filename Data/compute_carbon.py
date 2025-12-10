import pandas as pd
import numpy as np

def compute_co2_stats_per_day(
    csv_path="validation_co2_timeseries_all_rooms.csv",
    out_path="validation_co2_stats_by_day.csv",
):
    """
    Compare measured vs simulated CO2 per room & day:
    - mean, max, std for measured CO2
    - mean, max, std for simulated CO2
    - mean bias (sim - meas)
    - RMSE
    """
    df = pd.read_csv(csv_path)

    if "CO2_sim" not in df.columns:
        raise ValueError("Expected column 'CO2_sim' for simulated CO2.")

    # If there's no measured CO2, we can only report simulated stats
    has_measured = "CO2" in df.columns

    group_cols = ["room", "DayID"]
    stats_rows = []

    for (room, day_id), sub in df.groupby(group_cols):
        sub = sub.copy()

        # Simulated stats
        sim = sub["CO2_sim"].dropna()
        if sim.empty:
            continue  # nothing to do for this day

        sim_mean = sim.mean()
        sim_max = sim.max()
        sim_std = sim.std(ddof=1)  # sample std

        row = {
            "room": room,
            "DayID": day_id,
            "CO2_sim_mean": sim_mean,
            "CO2_sim_max": sim_max,
            "CO2_sim_std": sim_std,
        }

        if has_measured and not sub["CO2"].isna().all():
            meas = sub["CO2"].dropna()

            # align indices to compare point-wise (just in case some NaNs)
            joined = pd.DataFrame({"meas": sub["CO2"], "sim": sub["CO2_sim"]}).dropna()
            if not joined.empty:
                meas_mean = joined["meas"].mean()
                meas_max = joined["meas"].max()
                meas_std = joined["meas"].std(ddof=1)

                diff = joined["sim"] - joined["meas"]
                bias = diff.mean()                    # mean(sim - meas)
                rmse = np.sqrt((diff**2).mean())

                row.update({
                    "CO2_meas_mean": meas_mean,
                    "CO2_meas_max": meas_max,
                    "CO2_meas_std": meas_std,
                    "bias_sim_minus_meas": bias,
                    "rmse": rmse,
                    "n_points": len(joined),
                })
            else:
                # no overlapping non-NaN points
                row.update({
                    "CO2_meas_mean": np.nan,
                    "CO2_meas_max": np.nan,
                    "CO2_meas_std": np.nan,
                    "bias_sim_minus_meas": np.nan,
                    "rmse": np.nan,
                    "n_points": 0,
                })
        else:
            # no measured data available
            row.update({
                "CO2_meas_mean": np.nan,
                "CO2_meas_max": np.nan,
                "CO2_meas_std": np.nan,
                "bias_sim_minus_meas": np.nan,
                "rmse": np.nan,
                "n_points": len(sim),
            })

        stats_rows.append(row)

    df_stats = pd.DataFrame(stats_rows)
    df_stats.to_csv(out_path, index=False)
    print(f"Saved CO2 comparison stats to {out_path}")

    return df_stats




df_stats = compute_co2_stats_per_day()
print(df_stats.head())
df_stats.groupby("room")[["bias_sim_minus_meas", "rmse"]].mean()
